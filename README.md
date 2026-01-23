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

### Information Flow

```
                              ┌─────────────────────────────────┐
                              │        ADVERSARIAL LAYER        │
                              │                                 │
    ┌───────┐                 │  ┌───┐         ┌───┐           │
    │ Alice │──── message ────┼─►│ M │────┬───►│ D │──► clean? │
    └───────┘                 │  └───┘    │    └───┘           │
                              │   │       │      ▲             │
                              │   │inject │      │ original    │
                              │   ▼       │      │             │
                              └───┼───────┼──────┼─────────────┘
                                  │       │      │
                                  │       └──────┘
                                  ▼
                              ┌───────┐
                              │  Bob  │──► leak? ──► reward
                              └───────┘
```

D's task is purely classification—it sees both versions and must distinguish them. Bob (simulated) is the vulnerable agent that might leak information if M's injection is effective.

### Reward Structure

We use **GAN-style turn-based rewards** where the attacker (M) and defender (D) receive feedback each turn, plus terminal bonuses at episode end:

#### Per-Turn Rewards

| Event | R_M | R_D | Description |
|-------|-----|-----|-------------|
| D fooled | +1 | -1 | D classifies injected message as clean |
| D correct | 0 | +1 | D correctly classifies each message type |
| Injection succeeds | +1 | — | Flag leaked this turn |
| Injection fails | 0 | — | No penalty mid-episode (still exploring) |

#### Terminal Bonuses (Episode End)

| Outcome | R_M | R_D | Description |
|---------|-----|-----|-------------|
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
┌────────────────────────────────────────────────────────────────┐
│                     TRAINING ITERATION                         │
│                                                                │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │   Rollout    │      │   Compute    │      │    Update    │ │
│  │   Episodes   │─────►│  Advantages  │─────►│   M and D    │ │
│  │  (M vs D)    │      │   (GRPO)     │      │   (PPO-clip) │ │
│  └──────────────┘      └──────────────┘      └──────────────┘ │
│         │                                            │         │
│         │              ┌──────────────┐              │         │
│         └─────────────►│  Population  │◄─────────────┘         │
│                        │  (past M's)  │                        │
│                        └──────────────┘                        │
└────────────────────────────────────────────────────────────────┘
```

**Population-based attacker sampling** ensures the agent trains against diverse attack strategies, preventing overfitting to the current attacker's behavior.

## Project Structure

```
pigan/
├── src/pigan/
│   ├── envs/                    # Environment implementations
│   │   ├── env.py               # Main environment
│   │   ├── rewards.py           # Reward functions
│   │   ├── sandbox.py           # Code execution sandbox
│   │   └── scenarios/           # Task scenarios
│   └── utils/                   # Utilities
├── scripts/
│   ├── collect_demonstrations.py
│   ├── train_sft_warmup.py
│   ├── train_adversarial.py
│   └── run_pipeline.py
└── pyproject.toml
```

## Configuration

Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kl_coeff` | 0.01 | KL regularization strength for SFT |
| `episodes_per_iteration` | 16 | Rollouts per GRPO update |
| `group_size` | 4 | GRPO group size for advantage estimation |
| `max_turns` | 4 | Maximum turns per episode |

## License

MIT License
