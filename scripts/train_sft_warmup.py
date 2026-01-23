"""
SFT Warmup Training with KL Regularization.

This script implements imitation learning for warming up models before RL,
using ludic's training infrastructure with KL regularization.

Loss function:
    L_SFT(π) = E[ 1/|a| * Σ_i (-log π(a_i | s, a_{<i}) + β_SFT * KL[π || π_ref]) ]

Workflow:
    1. Collect demonstrations (collect_demonstrations.py) using a capable model
    2. Annotate with reference logprobs (this script --annotate mode)
    3. Train with SFT+KL (this script --train mode)

Usage:
    # Step 1: Annotate demonstrations with reference logprobs
    python train_sft_warmup.py annotate \
        --data demonstrations.jsonl \
        --model Qwen/Qwen2.5-3B-Instruct \
        --output demos_annotated.jsonl

    # Step 2: Train M (attacker)
    torchrun --nproc_per_node=2 train_sft_warmup.py train \
        --data demos_annotated.jsonl \
        --agent M \
        --kl-coeff 0.01 \
        --output-dir sft_warmup_m

    # Step 3: Train D (detector)
    torchrun --nproc_per_node=2 train_sft_warmup.py train \
        --data demos_annotated.jsonl \
        --agent D \
        --kl-coeff 0.01 \
        --output-dir sft_warmup_d
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch
import torch.distributed as dist
from torch.distributed import fsdp
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from ludic.types import Rollout, Step, AgentStep, EnvironmentStep, TokenTrace
from ludic.training import (
    OfflineBatchSource,
    Trainer,
    TrainerConfig,
    CheckpointConfig,
    make_sft_with_kl,
    make_sft,
    make_chat_template_step_to_item,
    SAWItem,
    SampleAttachments,
)
from ludic.training.types import ActorTokenLogps
from ludic.training.batching.annotate_logprobs import compute_reference_logprobs


# ---------------------------------------------------------------------------
# Data structures for demonstration samples
# ---------------------------------------------------------------------------


@dataclass
class DemonstrationSample:
    """A single demonstration sample for SFT (legacy format from collection)."""
    agent: Literal["M", "D"]
    prompt_token_ids: List[int]
    completion_token_ids: List[int]
    episode_id: str
    step_index: int
    reward: float
    injection_success: bool
    d_correct_on_original: bool
    d_correct_on_injected: bool
    d_fooled: bool
    flag_leaked: bool
    # Optional: reference logprobs (added during annotation)
    completion_logprobs: Optional[List[float]] = None


def load_demonstrations(path: str, agent: Optional[str] = None) -> List[DemonstrationSample]:
    """Load demonstration samples from JSONL, optionally filtered by agent."""
    samples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            sample = DemonstrationSample(**data)
            if agent is None or sample.agent == agent:
                samples.append(sample)
    return samples


def save_demonstrations(samples: List[DemonstrationSample], path: str) -> None:
    """Save demonstration samples to JSONL."""
    with open(path, "w") as f:
        for sample in samples:
            data = {
                "agent": sample.agent,
                "prompt_token_ids": sample.prompt_token_ids,
                "completion_token_ids": sample.completion_token_ids,
                "episode_id": sample.episode_id,
                "step_index": sample.step_index,
                "reward": sample.reward,
                "injection_success": sample.injection_success,
                "d_correct_on_original": sample.d_correct_on_original,
                "d_correct_on_injected": sample.d_correct_on_injected,
                "d_fooled": sample.d_fooled,
                "flag_leaked": sample.flag_leaked,
            }
            if sample.completion_logprobs is not None:
                data["completion_logprobs"] = sample.completion_logprobs
            f.write(json.dumps(data) + "\n")


# ---------------------------------------------------------------------------
# Convert to ludic rollout format
# ---------------------------------------------------------------------------


def demos_to_rollouts(
    samples: List[DemonstrationSample],
) -> List[Rollout]:
    """Convert demonstration samples to ludic Rollouts."""
    # Group by episode
    episodes: Dict[str, List[DemonstrationSample]] = {}
    for sample in samples:
        if sample.episode_id not in episodes:
            episodes[sample.episode_id] = []
        episodes[sample.episode_id].append(sample)

    rollouts = []
    for episode_id, ep_samples in episodes.items():
        # Sort by step index
        ep_samples.sort(key=lambda s: s.step_index)

        steps = []
        for sample in ep_samples:
            trace = TokenTrace(
                prompt_token_ids=sample.prompt_token_ids,
                completion_token_ids=sample.completion_token_ids,
                completion_logprobs=sample.completion_logprobs,
            )
            step = AgentStep(
                index=sample.step_index,
                prompt_messages=[],  # Not needed - we use token traces directly
                action="",  # Not needed - we use token traces
                action_target="env",
                loop_index=0,
                reward=sample.reward,
                truncated=False,
                terminated=(sample.step_index == ep_samples[-1].step_index),
                info={
                    "agent": sample.agent,
                    "injection_success": sample.injection_success,
                    "d_correct_on_original": sample.d_correct_on_original,
                    "d_correct_on_injected": sample.d_correct_on_injected,
                    "d_fooled": sample.d_fooled,
                    "flag_leaked": sample.flag_leaked,
                },
                trace=trace,
            )
            steps.append(step)

        rollout = Rollout(
            id=episode_id,
            steps=steps,
            meta={"agent": ep_samples[0].agent},
        )
        rollouts.append(rollout)

    return rollouts


def make_token_trace_step_to_item(
    *,
    extract_ref_logprobs: bool = False,
):
    """
    Create a step_to_item function that uses token traces directly.

    This is for demonstration data where we already have tokenized prompt/completion.
    """
    def step_to_item(rollout: Rollout, step: Step, weight: float) -> SAWItem:
        trace = step.trace
        if trace is None:
            raise ValueError(f"Step missing trace for rollout {rollout.id}, step {step.index}")

        # Build from token traces
        input_ids = list(trace.prompt_token_ids) + list(trace.completion_token_ids)
        attention_mask = [1] * len(input_ids)
        action_mask = [0] * len(trace.prompt_token_ids) + [1] * len(trace.completion_token_ids)

        meta = {
            "rollout_id": rollout.id,
            "step_index": step.index,
            "reward": float(step.reward),
            "total_reward": rollout.total_reward,
            "completion_length": len(trace.completion_token_ids),
            "prompt_length": len(trace.prompt_token_ids),
            **(step.info or {}),
        }

        # Extract reference logprobs if available
        attachments = SampleAttachments()
        if extract_ref_logprobs and trace.completion_logprobs is not None:
            if len(trace.completion_logprobs) != len(trace.completion_token_ids):
                raise ValueError(
                    f"completion_logprobs length ({len(trace.completion_logprobs)}) != "
                    f"completion_token_ids length ({len(trace.completion_token_ids)})"
                )
            attachments = SampleAttachments(
                actor_logps=ActorTokenLogps(token_logps=list(trace.completion_logprobs))
            )

        return SAWItem(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            weight=weight,
            meta=meta,
            attachments=attachments,
        )

    return step_to_item


# ---------------------------------------------------------------------------
# Distributed initialization
# ---------------------------------------------------------------------------


def init_dist(local_rank: int) -> int:
    """Initialize distributed training."""
    if dist.is_initialized():
        return dist.get_rank()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    else:
        dist.init_process_group(backend="gloo", init_method="env://")
    return dist.get_rank()


# ---------------------------------------------------------------------------
# Annotate command
# ---------------------------------------------------------------------------


def cmd_annotate(args: argparse.Namespace) -> None:
    """Annotate demonstration data with reference policy logprobs."""
    print(f"Loading demonstrations from: {args.data}")
    samples = load_demonstrations(args.data, agent=args.agent)
    print(f"Loaded {len(samples)} samples" + (f" for agent {args.agent}" if args.agent else ""))

    if not samples:
        print("No samples to annotate!")
        return

    print(f"Loading reference model: {args.model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    print(f"Computing reference logprobs for {len(samples)} samples...")

    # Process in batches
    batch_size = args.batch_size
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]

        # Find max length for padding
        max_len = max(len(s.prompt_token_ids) + len(s.completion_token_ids) for s in batch)

        # Prepare tensors
        pad_token_id = 0  # Most tokenizers use 0
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        action_mask = torch.zeros((len(batch), max_len), dtype=torch.float32)

        for b, sample in enumerate(batch):
            full_ids = sample.prompt_token_ids + sample.completion_token_ids
            L = len(full_ids)
            input_ids[b, :L] = torch.tensor(full_ids, dtype=torch.long)
            attention_mask[b, :L] = 1
            action_mask[b, len(sample.prompt_token_ids):L] = 1.0

        # Compute logprobs
        ref_logprobs = compute_reference_logprobs(
            model, input_ids, attention_mask, action_mask, device=device
        )

        # Store in samples
        for b, sample in enumerate(batch):
            sample.completion_logprobs = ref_logprobs[b]

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch)}/{len(samples)} samples")

    print(f"Saving annotated samples to: {args.output}")
    save_demonstrations(samples, args.output)
    print("Done!")


# ---------------------------------------------------------------------------
# Train command
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    """Train with SFT+KL using ludic infrastructure."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = init_dist(local_rank)
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # Load demonstrations
    data_path = Path(args.data)
    if not data_path.exists():
        if rank == 0:
            print(f"Error: Data file not found: {data_path}")
        dist.destroy_process_group()
        return

    if rank == 0:
        print(f"Loading demonstrations for agent {args.agent}...")

    samples = load_demonstrations(str(data_path), agent=args.agent)

    if rank == 0:
        print(f"Loaded {len(samples)} samples")

    if not samples:
        if rank == 0:
            print(f"No samples found for agent {args.agent}!")
        dist.destroy_process_group()
        return

    # Check if samples have reference logprobs
    has_logprobs = all(s.completion_logprobs is not None for s in samples)
    if args.kl_coeff > 0 and not has_logprobs:
        if rank == 0:
            print("Warning: KL coefficient > 0 but samples don't have reference logprobs.")
            print("Run 'python train_sft_warmup.py annotate' first, or set --kl-coeff 0")
        dist.destroy_process_group()
        return

    # Convert to rollouts and save as temp JSONL
    rollouts = demos_to_rollouts(samples)
    temp_jsonl = Path(args.output_dir) / "temp_rollouts.jsonl"
    temp_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        with open(temp_jsonl, "w") as f:
            for rollout in rollouts:
                f.write(json.dumps(rollout.to_dict()) + "\n")

    dist.barrier()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # FSDP2 mixed precision policy
    mp_policy = fsdp.MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    # Load model
    if rank == 0:
        print(f"Loading model: {args.model}")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Apply LoRA if requested
    if args.use_lora:
        if rank == 0:
            print(f"Applying LoRA (rank={args.lora_rank})")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=int(args.lora_rank * args.lora_alpha_mult),
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules="all-linear",
        )
        model = get_peft_model(base_model, lora_config)
        if rank == 0:
            model.print_trainable_parameters()
    else:
        model = base_model

    # Apply FSDP2 sharding
    blocks = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    elif hasattr(model, "layers"):
        blocks = model.layers
    if blocks is not None:
        for layer in blocks:
            fsdp.fully_shard(layer, mp_policy=mp_policy)
    fsdp.fully_shard(model, mp_policy=mp_policy)

    # Create algorithm
    if args.kl_coeff > 0:
        algo = make_sft_with_kl(
            kl_coeff=args.kl_coeff,
            length_normalize=True,
        )
    else:
        algo = make_sft(length_normalize=True)

    # Create step_to_item function
    step_to_item = make_token_trace_step_to_item(
        extract_ref_logprobs=(args.kl_coeff > 0),
    )

    # Create batch source
    batch_source = OfflineBatchSource(
        jsonl_paths=[temp_jsonl],
        step_to_item=step_to_item,
        credit_assigner=algo.credit_assigner,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Calculate training steps
    batches_per_epoch = batch_source.num_batches_per_epoch
    total_steps = args.epochs * batches_per_epoch

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"SFT Warmup - Agent {args.agent}")
        print(f"{'='*60}")
        print(f"Model: {args.model}")
        print(f"LoRA: {'enabled' if args.use_lora else 'disabled'}")
        print(f"KL coefficient (β_SFT): {args.kl_coeff}")
        print(f"World size: {world_size}")
        print(f"Samples: {len(samples)}")
        print(f"Batches per epoch: {batches_per_epoch}")
        print(f"Total steps: {total_steps} ({args.epochs} epochs)")
        print(f"{'='*60}\n")

    # Trainer config
    cfg = TrainerConfig(
        model_device=str(device),
        max_seq_len=args.max_seq_len,
        micro_token_budget=args.micro_token_budget,
        max_grad_norm=args.max_grad_norm,
        pad_token_id=tokenizer,
        lr=args.lr,
        reduce_stats_across_ranks=True,
        eval_at_start=False,
        eval_every_n_steps=None,
        sync_every_steps=0,
    )

    checkpoint_cfg = None
    if args.checkpoint_every > 0:
        checkpoint_cfg = CheckpointConfig(
            output_dir=args.output_dir,
            every_n_steps=args.checkpoint_every,
            max_to_keep=args.max_to_keep,
            save_optimizer=True,
        )
        if rank == 0:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        dist.barrier()

    trainer = Trainer(
        model=model,
        algo=algo,
        batch_source=batch_source,
        publisher=None,
        cfg=cfg,
        checkpoint_config=checkpoint_cfg,
        enable_gradient_checkpointing=bool(args.gradient_checkpointing),
        train_logger=None,
        evaluator=None,
    )

    async def train_loop() -> None:
        for step in range(total_steps):
            stats = await trainer.train_step()
            if rank == 0 and (step % args.log_every == 0):
                loss = stats.get("train/loss", float("nan"))
                if args.kl_coeff > 0:
                    sft_loss = stats.get("train/sft/loss", float("nan"))
                    kl_loss = stats.get("train/kl/loss", float("nan"))
                    kl_mean = stats.get("train/kl/kl_mean", float("nan"))
                    print(
                        f"[step {step + 1}/{total_steps}] "
                        f"loss={loss:.4f} sft={sft_loss:.4f} kl={kl_loss:.4f} "
                        f"kl_mean={kl_mean:.4f}",
                        flush=True,
                    )
                else:
                    logp = stats.get("train/logp_mean", float("nan"))
                    print(
                        f"[step {step + 1}/{total_steps}] loss={loss:.4f} logp={logp:.4f}",
                        flush=True,
                    )

    asyncio.run(train_loop())

    if rank == 0:
        print("\nTraining complete!")

        # Save final model
        if args.final_save:
            final_path = Path(args.output_dir) / "final"
            if args.use_lora:
                # Save LoRA adapter
                model.save_pretrained(final_path)
            else:
                # For full model, we'd need FSDP state_dict handling
                pass
            tokenizer.save_pretrained(final_path)
            print(f"Saved final model to: {final_path}")

        # Save training info
        info_path = Path(args.output_dir) / "training_info.json"
        with open(info_path, "w") as f:
            json.dump({
                "agent": args.agent,
                "model": args.model,
                "num_samples": len(samples),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "kl_coeff": args.kl_coeff,
                "use_lora": args.use_lora,
                "lora_rank": args.lora_rank if args.use_lora else None,
            }, f, indent=2)

        # Cleanup temp file
        if temp_jsonl.exists():
            temp_jsonl.unlink()

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="SFT Warmup Training with KL Regularization"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Annotate command
    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Annotate demonstrations with reference logprobs"
    )
    annotate_parser.add_argument("--data", required=True, help="Input demonstrations JSONL")
    annotate_parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                                  help="Reference model for logprobs")
    annotate_parser.add_argument("--output", required=True, help="Output annotated JSONL")
    annotate_parser.add_argument("--agent", choices=["M", "D"], default=None,
                                  help="Filter for specific agent (optional)")
    annotate_parser.add_argument("--batch-size", type=int, default=8)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train with SFT+KL")
    train_parser.add_argument("--data", required=True, help="Annotated demonstrations JSONL")
    train_parser.add_argument("--agent", choices=["M", "D"], required=True,
                               help="Which agent to train")
    train_parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    train_parser.add_argument("--output-dir", required=True)

    # KL regularization
    train_parser.add_argument("--kl-coeff", type=float, default=0.01,
                               help="KL regularization coefficient (β_SFT). 0 = pure SFT")

    # LoRA
    train_parser.add_argument("--use-lora", action="store_true", default=True)
    train_parser.add_argument("--no-lora", action="store_false", dest="use_lora")
    train_parser.add_argument("--lora-rank", type=int, default=8)
    train_parser.add_argument("--lora-alpha-mult", type=float, default=2.0)
    train_parser.add_argument("--lora-dropout", type=float, default=0.0)

    # Training
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=2e-5)
    train_parser.add_argument("--max-seq-len", type=int, default=2048)
    train_parser.add_argument("--micro-token-budget", type=int, default=8192)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)

    # Checkpointing
    train_parser.add_argument("--checkpoint-every", type=int, default=100)
    train_parser.add_argument("--max-to-keep", type=int, default=2)
    train_parser.add_argument("--final-save", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--log-every", type=int, default=1)

    args = parser.parse_args()

    if args.command == "annotate":
        cmd_annotate(args)
    elif args.command == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
