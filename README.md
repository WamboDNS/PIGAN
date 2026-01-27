# PiGAN - Adversarial Prompt Injection Training

Train LLMs to discover prompt injection vulnerabilities through adversarial reinforcement learning.

## Overview

This repository contains a [PrimeIntellect verifiers](https://github.com/PrimeIntellect-ai/verifiers) environment for training prompt injection attackers. An attacker model (Adv) learns to craft malicious instructions that trick a target LLM into leaking secrets.

## Environment

The environment is located at [`environments/injection_trainer/`](environments/injection_trainer/).

### Quick Start

```bash
# Run evaluation
prime eval run injection-trainer

# With configuration
prime eval run injection-trainer \
  -m gpt-4.1-mini \
  -a '{"attack_modes": ["memory"], "defense_levels": ["easy"]}'
```

### How It Works

```
Adv (trained) → injects into user message → Target (frozen) → rewards
     ↑                                              │
     └──────── refine based on response ←───────────┘
```

1. Adv receives a user prompt and outputs `<injection>...</injection>`
2. Environment combines injection with user message, sends to Target
3. Target responds (may use tools in sandbox mode)
4. Rewards computed based on whether secret was leaked
5. Multi-turn: Adv can refine attacks based on Target's response

### Attack Modes

| Mode | Description |
|------|-------------|
| `memory` | Secret in target's system prompt |
| `sandbox` | Secret in file, target has `read_file`/`list_directory` tools |

### Defense Levels

| Level | Description |
|-------|-------------|
| `easy` | Basic "don't share" instruction |
| `medium` | Explicit security warnings |
| `hard` | Multi-rule defense with pattern detection |

## Use Cases

- **Red-teaming**: Test LLM robustness against injection attacks
- **Attack generation**: Create diverse attack patterns for defensive training
- **Security research**: Study prompt injection vulnerabilities

## License

MIT
