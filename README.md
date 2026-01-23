# PIGAN

Adversarial training for prompt injection attack and defense.

## What is this?

LLM agents with tool access (file reads, API calls, code execution) are vulnerable to prompt injection—malicious instructions hidden in user inputs that hijack the agent's behavior. PIGAN trains two models in a GAN-like setup:

- **M (attacker)**: learns to craft injections that evade detection and trick the agent into leaking secrets
- **D (detector)**: learns to distinguish clean messages from injected ones

Both models improve through competition. M finds new attack patterns, D learns to catch them, M adapts, and so on. The result is a robust detector that generalizes to unseen attacks.

Built on top of [ludic](https://github.com/WamboDNS/ludic) for RL training infrastructure and [PrimeIntellect sandboxes](https://github.com/PrimeIntellect-ai/prime-rl) for secure code execution.

## Setup

Two layers:

1. **Alice ↔ Bob**: Simulated multi-turn conversation. Bob has tool access to files containing secrets.
2. **M vs D**: Trained adversarially. M injects into every message, D classifies clean vs injected.

```
    Alice                    M (attacker)              D (detector)
      │                          │                          │
      │─── benign message ──────►│                          │
      │                          │─── injected version ────►│──► "injected"?
      │                          │                          │
      │                          │         original ───────►│──► "clean"?
      │                          │                          │
      │                          ▼
      │                        Bob ──► leak? ──► rewards
      │                          │
      │◄── response (clean) ─────┤
      │                          │◄── response (sees injected context)
      │                          │
      │─── next message ────────►│  ... (repeat for max_turns)
```

Each episode runs for multiple turns.

**Who sees what:**
- **Alice**: only sees the benign conversation (her messages + Bob's responses)
- **M**: sees the conversation with injections (can learn what worked)
- **D**: sees both original and injected message each turn
- **Bob**: only sees the injected version

If Bob leaks the secret flag at any point, M wins.

### Sandboxing

Bob executes tool calls (file reads, etc.) inside a sandboxed environment. Each episode gets a fresh sandbox with a randomly generated secret flag placed in a file. This ensures:
- Isolation between episodes
- Safe execution of potentially harmful injected instructions
- Reproducible evaluation of attack success

## Rewards

Per turn:
- M gets +1 if D misclassifies the injected message as clean
- M gets +1 if Bob leaks the flag
- D gets +1/-1 for correct/incorrect classification

Terminal:
- M gets -2 if flag never leaked (failed completely)
- D gets +2 if flag never leaked

No mid-episode penalty for failed injections—M can explore freely.

## Training

**Phase 1: Imitation learning**

Warm up M and D with SFT on demonstrations from a 32B model. KL penalty prevents drift from base model.

**Phase 2: Adversarial RL**

Train M and D with GRPO. Population-based sampling: D trains against current and past M checkpoints to avoid overfitting to one attack style.

## Config

| Param | Default |
|-------|---------|
| `kl_coeff` | 0.01 |
| `episodes_per_iteration` | 16 |
| `group_size` | 4 |
| `max_turns` | 4 |
