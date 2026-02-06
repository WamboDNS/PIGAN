# Multi-Agent Text Reversal Environment

A multi-agent RL environment where two agents cooperatively learn text reversal:

- **Agent A (reverser)**: Receives original text, outputs the reversed version
- **Agent B (de-reverser)**: Receives Agent A's output, reverses it back

Both agents are independently trainable with LCS-based rewards.

## Architecture

```
Original Text: "hello"
     │
     ▼
┌─────────────┐
│  Reverser   │ ──► "olleh"  (Reward: LCS vs "olleh")
│   (Agent A) │
└─────────────┘
     │
     ▼
┌─────────────┐
│ De-Reverser │ ──► "hello"  (Reward: LCS vs "hello")
│   (Agent B) │
└─────────────┘
```

## Reward Design

Each agent has independent LCS-ratio rewards:

- **Reverser reward**: `LCS(parsed_output, original[::-1])`
- **De-reverser reward**: `LCS(parsed_output, reverser_output[::-1])`
  - Note: Independent of original text - only depends on what Agent A produced

## Usage

```python
from reverse_multi_agent import load_environment

env = load_environment(
    dataset_name="PrimeIntellect/Reverse-Text-RL",
    dataset_split="train",
)
```

## Configuration

- Uses `<reversed_text>` XML tags for parsing outputs
- Sequential turn order: reverser → de-reverser
- Both agents use separate LoRA adapters (lora_id 0 and 1)
