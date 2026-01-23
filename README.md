# PIGAN: Prompt Injection GAN

GAN-style adversarial training for prompt injection attack and defense in LLM agents.

## Problem Setting

Modern LLM-based agents interact with external tools, APIs, and user data—creating attack surfaces for **prompt injection**. An adversary can embed malicious instructions in seemingly benign inputs, attempting to hijack the agent's behavior and leak sensitive information.

### Two-Layer Architecture

The environment has two layers:

**Layer 1: Benign Conversation (Simulated)**
- **Alice**: User who sends benign messages
- **Bob**: Assistant with tool access (file read, etc.) that may contain sensitive data

**Layer 2: Adversarial Game (Trained)**
- **M (Attacker)**: Intercepts Alice's messages and injects malicious instructions into EVERY message
- **D (Detector)**: Classifies messages as "clean" or "injected"

### Turn Flow

```
1. Alice sends a benign message
2. M ALWAYS injects into Alice's message (creates manipulated version)
3. D classifies the ORIGINAL message (ground truth: clean)
4. D classifies the INJECTED message (ground truth: injected)
5. Bob receives M's injected version and responds
6. Check if Bob leaked the secret flag → injection success
7. Compute rewards, continue or terminate
```

D's task is purely classification—it sees both versions and must distinguish them. Bob (simulated) is the vulnerable agent that might leak information if M's injection is effective.

### Reward Structure

We use **GAN-style turn-based rewards** where the attacker (M) and defender (D) receive feedback each turn, plus terminal bonuses at episode end:

#### Per-Turn Rewards

| Event | R_atk | R_agt | Description |
|-------|-------|-------|-------------|
| D fooled | +1 | -1 | D classifies injected message as clean |
| D correct | 0 | +1 | D correctly classifies each message type |
| Injection succeeds | +1 | — | Flag leaked this turn |
| Injection fails | 0 | — | No penalty mid-episode (still exploring) |

#### Terminal Bonuses (Episode End)

| Outcome | R_atk | R_agt | Description |
|---------|-------|-------|-------------|
| Flag leaked at any point | +1 | -1 | Attacker succeeded overall |
| Flag never leaked | -2 | +2 | Attacker failed completely |

This structure encourages:
- **M** to keep exploring strategies without heavy mid-episode penalties
- **M** to eventually succeed before episode ends (or face -2 terminal penalty)
- **D** to be vigilant every turn with immediate feedback

### Zero-Sum Adversarial Dynamics

Throughout training, M and D are **simultaneously optimized** in a minimax game:
- **M** learns increasingly sophisticated injection techniques that evade detection
- **D** learns to distinguish clean messages from injected ones
- This co-evolution produces robust detectors that generalize to unseen attack patterns

## Training Pipeline

### Phase 1: Imitation Learning (Warmup)

Due to limited initial capabilities, base models have low attack success rates and task success rates, leading to inefficient exploration during RL. We first warm up the models using **imitation learning with KL regularization**.

1. **Collect demonstrations** using a capable model (32B+ parameters recommended)
2. **Filter for successful episodes**:
   - Attacker: episodes where information was leaked
   - Agent: episodes where task succeeded without leaks
3. **Compute reference logprobs** from the base model to be fine-tuned
4. **Train with SFT + KL regularization**:

```
L_SFT(π) = E[ (1/|a|) * Σᵢ ( -log π(aᵢ|s, a_{<i}) + β * KL_token ) ]
```

where:
- First term: standard supervised fine-tuning (imitate demonstrations)
- Second term: KL penalty to prevent drift from the base model
- β: regularization strength (typically 0.01 - 0.1)

### Phase 2: Adversarial Reinforcement Learning

After warmup, we train both policies using **GRPO** (Group Relative Policy Optimization):

```
Algorithm: Adversarial RL Training
────────────────────────────────────────────────────────────────
1. Initialize attacker π_atk, agent π_agt from warmup checkpoints
2. Initialize population P_atk ← {π_atk}  // attacker checkpoint population

3. For iteration = 1 to N_iter:
   4. For episode = 1 to N_episodes:
      5. Sample attacker π̃_atk ~ P_atk     // population sampling
      6. For step t = 1 to T:
         7. Attacker generates injection: a_atk = π̃_atk(s)
         8. Agent observes: s_agt = (s, s_user, a_atk)
         9. Agent selects action: a_agt = π_agt(s_agt)
         10. Environment transitions: s' = T(s, a_atk, a_agt)
         11. Store transition: (s, a_atk, a_agt, r_atk, r_agt, s')
      12. End For
   13. End For
   14. Update π_atk and π_agt using GRPO
   15. P_atk ← P_atk ∪ {π_atk}              // add checkpoint to population
16. End For
17. Return final agent π_agt
────────────────────────────────────────────────────────────────
```

**Population-based attacker sampling** (step 5) ensures the agent trains against diverse attack strategies, preventing overfitting to the current attacker's behavior.

## Installation

```bash
# Clone the repository
git clone https://github.com/youruser/pigan.git
cd pigan

# Install with ludic dependency
pip install -e .

# Or for development with local ludic
pip install -e /path/to/ludic -e .
```

## Quick Start

### 1. Collect Demonstrations

```bash
# Start vLLM server with teacher model
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2

# Collect demonstrations
python scripts/collect_demonstrations.py \
    --teacher-model Qwen/Qwen2.5-32B-Instruct \
    --num-episodes 500 \
    --output data/demonstrations.jsonl
```

### 2. Imitation Learning Warmup

```bash
# Annotate with reference logprobs
python scripts/train_sft_warmup.py annotate \
    --data data/demonstrations.jsonl \
    --model Qwen/Qwen2.5-3B-Instruct \
    --output data/demonstrations_annotated.jsonl

# Train attacker (M)
torchrun --nproc_per_node=2 scripts/train_sft_warmup.py train \
    --data data/demonstrations_annotated.jsonl \
    --agent M \
    --kl-coeff 0.01 \
    --output-dir checkpoints/warmup_m

# Train agent (D)
torchrun --nproc_per_node=2 scripts/train_sft_warmup.py train \
    --data data/demonstrations_annotated.jsonl \
    --agent D \
    --kl-coeff 0.01 \
    --output-dir checkpoints/warmup_d
```

### 3. Adversarial RL Training

```bash
torchrun --nproc_per_node=4 scripts/train_adversarial.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --m-warmup checkpoints/warmup_m/final \
    --d-warmup checkpoints/warmup_d/final \
    --output-dir checkpoints/adversarial \
    --num-iterations 100 \
    --episodes-per-iteration 64
```

### One-Command Pipeline

```bash
python scripts/run_pipeline.py \
    --teacher-model Qwen/Qwen2.5-32B-Instruct \
    --student-model Qwen/Qwen2.5-3B-Instruct \
    --num-episodes 500 \
    --output-dir ./training_run
```

## Project Structure

```
adversarial-robustness/
├── src/adversarial_robustness/
│   ├── envs/                    # Environment implementations
│   │   ├── prompt_injection.py  # Main environment
│   │   ├── rewards.py           # Reward functions
│   │   ├── sandbox.py           # Code execution sandbox
│   │   └── scenarios/           # Task scenarios
│   └── utils/                   # Utilities
├── scripts/
│   ├── collect_demonstrations.py
│   ├── train_sft_warmup.py
│   ├── train_adversarial.py
│   └── run_pipeline.py
├── configs/                     # Configuration files
├── tests/                       # Unit tests
└── pyproject.toml
```

## Configuration

Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kl_coeff` | 0.01 | KL regularization strength for SFT |
| `episodes_per_iteration` | 64 | Rollouts per GRPO update |
| `group_size` | 4 | GRPO group size for advantage estimation |
| `max_steps` | 20 | Maximum steps per episode (L) |
| `population_size` | 10 | Attacker checkpoint population size |

## Citation

```bibtex
@article{adversarial-robustness-2025,
  title={Adversarial Robustness Training for LLM Agents},
  author={...},
  year={2025}
}
```

## License

MIT License
