#!/usr/bin/env python3
"""
Imitation Learning Pipeline.

This script automates the complete imitation learning workflow:
1. Collect demonstrations using a capable teacher model (32B+ recommended)
2. Annotate with reference logprobs from the base model
3. Train M (attacker) and D (detector) with SFT+KL

Loss function:
    L_SFT(π) = E[ 1/|a| * Σ_i (-log π(a_i | s, a_{<i}) + β_SFT * KL[π || π_ref]) ]

Usage:
    # Full pipeline (requires vLLM server running)
    python run_sft_pipeline.py \
        --teacher-model Qwen/Qwen2.5-14B-Instruct \
        --student-model Qwen/Qwen2.5-3B-Instruct \
        --num-episodes 500 \
        --output-dir ./sft_warmup

    # Skip collection (use existing demonstrations)
    python run_sft_pipeline.py \
        --skip-collect \
        --demos demonstrations.jsonl \
        --student-model Qwen/Qwen2.5-3B-Instruct \
        --output-dir ./sft_warmup

    # Only collect and annotate (no training)
    python run_sft_pipeline.py \
        --teacher-model Qwen/Qwen2.5-14B-Instruct \
        --student-model Qwen/Qwen2.5-3B-Instruct \
        --num-episodes 500 \
        --skip-train \
        --output-dir ./sft_warmup
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return the exit code."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with exit code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Imitation Learning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python run_sft_pipeline.py \\
      --teacher-model Qwen/Qwen2.5-14B-Instruct \\
      --student-model Qwen/Qwen2.5-3B-Instruct \\
      --num-episodes 500

  # Use existing demonstrations
  python run_sft_pipeline.py \\
      --skip-collect \\
      --demos existing_demos.jsonl \\
      --student-model Qwen/Qwen2.5-3B-Instruct
        """
    )

    # Pipeline control
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip demonstration collection (use --demos instead)")
    parser.add_argument("--skip-annotate", action="store_true",
                        help="Skip logprob annotation (demos already have logprobs)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (only collect and annotate)")
    parser.add_argument("--train-m-only", action="store_true",
                        help="Only train M (attacker)")
    parser.add_argument("--train-d-only", action="store_true",
                        help="Only train D (detector)")

    # Models
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-32B-Instruct",
                        help="Teacher model for demonstrations (32B+ recommended)")
    parser.add_argument("--student-model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="Student model to train (base model)")

    # Collection
    parser.add_argument("--demos", type=str, default=None,
                        help="Existing demonstrations file (with --skip-collect)")
    parser.add_argument("--num-episodes", type=int, default=200,
                        help="Number of episodes to collect")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"],
                        default="easy")
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)

    # Training
    parser.add_argument("--kl-coeff", type=float, default=0.01,
                        help="KL regularization coefficient (β_SFT)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs for training")

    # Output
    parser.add_argument("--output-dir", type=str, default="./sft_warmup",
                        help="Output directory for all artifacts")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    raw_demos = output_dir / "demonstrations_raw.jsonl"
    annotated_demos = output_dir / "demonstrations_annotated.jsonl"
    m_output_dir = output_dir / "model_m"
    d_output_dir = output_dir / "model_d"

    script_dir = Path(__file__).parent

    # -------------------------------------------------------------------------
    # Step 1: Collect demonstrations
    # -------------------------------------------------------------------------
    if not args.skip_collect:
        if args.demos:
            print(f"Warning: --demos ignored when not using --skip-collect")

        collect_cmd = [
            sys.executable, str(script_dir / "collect_demonstrations.py"),
            "--teacher-model", args.teacher_model,
            "--host", args.vllm_host,
            "--port", str(args.vllm_port),
            "--num-episodes", str(args.num_episodes),
            "--difficulty", args.difficulty,
            "--output", str(raw_demos),
            "--filter-agent", "both",
        ]

        ret = run_command(collect_cmd, "Collecting demonstrations")
        if ret != 0:
            sys.exit(ret)

        demos_file = raw_demos
    else:
        if not args.demos:
            print("ERROR: --demos required when using --skip-collect")
            sys.exit(1)
        demos_file = Path(args.demos)
        if not demos_file.exists():
            print(f"ERROR: Demonstrations file not found: {demos_file}")
            sys.exit(1)
        print(f"Using existing demonstrations: {demos_file}")

    # -------------------------------------------------------------------------
    # Step 2: Annotate with reference logprobs
    # -------------------------------------------------------------------------
    if not args.skip_annotate:
        annotate_cmd = [
            sys.executable, str(script_dir / "train_sft_warmup.py"), "annotate",
            "--data", str(demos_file),
            "--model", args.student_model,
            "--output", str(annotated_demos),
        ]

        ret = run_command(annotate_cmd, "Annotating with reference logprobs")
        if ret != 0:
            sys.exit(ret)

        training_demos = annotated_demos
    else:
        training_demos = demos_file
        print(f"Skipping annotation, using: {training_demos}")

    # -------------------------------------------------------------------------
    # Step 3: Train M and D
    # -------------------------------------------------------------------------
    if args.skip_train:
        print("\n" + "="*60)
        print("Pipeline complete (training skipped)")
        print("="*60)
        print(f"Demonstrations: {training_demos}")
        sys.exit(0)

    # Build training command template
    def make_train_cmd(agent: str, out_dir: Path) -> list[str]:
        if args.gpus > 1:
            cmd = [
                "torchrun",
                f"--nproc_per_node={args.gpus}",
                str(script_dir / "train_sft_warmup.py"), "train",
            ]
        else:
            cmd = [
                sys.executable, str(script_dir / "train_sft_warmup.py"), "train",
            ]

        cmd += [
            "--data", str(training_demos),
            "--agent", agent,
            "--model", args.student_model,
            "--output-dir", str(out_dir),
            "--kl-coeff", str(args.kl_coeff),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--lora-rank", str(args.lora_rank),
        ]
        return cmd

    # Train M (attacker)
    if not args.train_d_only:
        train_m_cmd = make_train_cmd("M", m_output_dir)
        ret = run_command(train_m_cmd, "Training M (attacker)")
        if ret != 0:
            print("Warning: M training failed, continuing...")

    # Train D (detector)
    if not args.train_m_only:
        train_d_cmd = make_train_cmd("D", d_output_dir)
        ret = run_command(train_d_cmd, "Training D (detector)")
        if ret != 0:
            print("Warning: D training failed, continuing...")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Imitation Learning Pipeline Complete!")
    print("="*60)
    print(f"\nArtifacts:")
    print(f"  Demonstrations: {training_demos}")
    if not args.train_d_only and m_output_dir.exists():
        print(f"  M model: {m_output_dir / 'final'}")
    if not args.train_m_only and d_output_dir.exists():
        print(f"  D model: {d_output_dir / 'final'}")

    print(f"\nTo use in adversarial RL training:")
    print(f"  python train_adversarial.py \\")
    if not args.train_d_only:
        print(f"      --m-warmup {m_output_dir / 'final'} \\")
    if not args.train_m_only:
        print(f"      --d-warmup {d_output_dir / 'final'} \\")
    print(f"      --model {args.student_model}")


if __name__ == "__main__":
    main()
