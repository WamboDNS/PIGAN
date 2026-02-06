"""
Multi-Agent Text Reversal Environment.

Two agents:
- Agent A (reverser): Takes input text and reverses it
- Agent B (de-reverser): Takes Agent A's output and reverses it back

Both agents are trainable with independent LCS-based rewards.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset
from verifiers.agents.agent import AgentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def lcs_ratio(x: str, y: str) -> float:
    """Return the longest common subsequence ratio of x and y."""
    return SequenceMatcher(None, x, y).ratio()


# =============================================================================
# Prompts
# =============================================================================


REVERSER_SYSTEM_PROMPT = """You are a text reversal assistant.
Your task is to reverse the given text character-by-character.
Put your answer in <reversed_text> tags."""

DE_REVERSER_SYSTEM_PROMPT = """You are a text reversal assistant.
Your task is to reverse the given text character-by-character.
Put your answer in <reversed_text> tags."""


# =============================================================================
# Reward Functions
# =============================================================================


async def reverser_reward(completion: vf.Messages, state: vf.State) -> float:
    """
    Reward for Agent A (reverser).
    LCS ratio between parsed output and correct reversal of original text.
    """
    parser = state.get("_parser")
    original_text = state.get("original_text", "")
    target = original_text[::-1]

    if not completion or not parser:
        return 0.0

    # Get the agent's output from completion (provided by MultiAgentRubric)
    content = (completion[-1].get("content", "") or "").strip()
    parsed = parser.parse_answer(content) or ""
    return lcs_ratio(parsed, target)


async def de_reverser_reward(completion: vf.Messages, state: vf.State) -> float:
    """
    Reward for Agent B (de-reverser).
    LCS ratio between parsed output and reversal of Agent A's output.
    Independent of the original text - just reverse what A produced.
    """
    parser = state.get("_parser")
    reverser_parsed = state.get("_reverser_parsed", "")

    if not completion or not parser:
        return 0.0

    # Target is the reversal of what reverser produced
    target = reverser_parsed[::-1]

    # Get the agent's output from completion (provided by MultiAgentRubric)
    content = (completion[-1].get("content", "") or "").strip()
    parsed = parser.parse_answer(content) or ""
    return lcs_ratio(parsed, target)


# =============================================================================
# Multi-Agent Environment
# =============================================================================


class ReverseMultiAgentEnv(vf.MultiAgentEnv):
    """
    Multi-agent environment for text reversal training.

    Two agents:
    - reverser: Takes original text, outputs reversal
    - de_reverser: Takes reverser's output, outputs its reversal (independent of original)

    Both agents are trainable with LCS-based rewards.
    """

    def __init__(
        self,
        dataset: Dataset,
        rubric: vf.Rubric,
        **kwargs: Any,
    ):
        # Create agent configs - both trainable with separate LoRA adapters
        agents = {
            "reverser": AgentConfig(
                agent_id="reverser",
                system_prompt=REVERSER_SYSTEM_PROMPT,
                trainable=True,
                lora_id=0,
            ),
            "de_reverser": AgentConfig(
                agent_id="de_reverser",
                system_prompt=DE_REVERSER_SYSTEM_PROMPT,
                trainable=True,
                lora_id=1,
            ),
        }

        # Parser for extracting answers from XML tags
        self._parser = vf.XMLParser(["reversed_text"], answer_field="reversed_text")

        super().__init__(
            agents=agents,
            turn_order="sequential",
            max_turns=2,  # One turn each: reverser then de_reverser
            dataset=dataset,
            rubric=rubric,
            parser=self._parser,
            **kwargs,
        )

    async def get_active_agents(self, state: vf.State) -> list[str]:
        """Sequential: reverser first, then de_reverser."""
        current_turn = state.get("current_turn", 0)
        if current_turn == 0:
            return ["reverser"]
        else:
            return ["de_reverser"]

    async def get_initial_observation(
        self, agent_id: str, state: vf.State
    ) -> str | None:
        """
        Initial observation for each agent.
        - reverser: gets the original text to reverse
        - de_reverser: None initially (waits for reverser's output)
        """
        if agent_id == "reverser":
            original_text = state.get("original_text", "")
            return f"Please reverse this text: {original_text}"
        return None

    async def get_agent_observation(
        self, agent_id: str, response: vf.Messages, state: vf.State
    ) -> str | None:
        """
        Inter-agent communication.
        de_reverser receives reverser's parsed output.
        """
        acting_agent = state.get("_last_acting_agent")

        if acting_agent == "reverser" and agent_id == "de_reverser":
            # de_reverser receives reverser's parsed output
            reverser_parsed = state.get("_reverser_parsed", "")
            return f"Please reverse this text: {reverser_parsed}"

        return None

    async def setup_state(self, state: vf.State) -> vf.State:
        """Initialize episode state."""
        # Extract original text from dataset
        question = state.get("question", "")
        state["original_text"] = question

        # Store parser reference for reward functions
        state["_parser"] = self._parser

        # Initialize tracking
        state["_reverser_output"] = ""
        state["_reverser_parsed"] = ""
        state["_de_reverser_output"] = ""
        state["_last_acting_agent"] = None

        logger.info(
            f"[ReverseMultiAgentEnv.setup_state] example_id={state.get('example_id')} "
            f"original_text='{question[:50]}...'"
        )

        state = await super().setup_state(state)
        return state

    async def _agent_turn(
        self,
        agent_id: str,
        state: vf.State,
    ) -> vf.TrajectoryStep:
        """Execute a turn and capture outputs for reward computation."""
        step = await super()._agent_turn(agent_id, state)

        state["_last_acting_agent"] = agent_id

        completion = step.get("completion", [])
        if completion:
            content = (completion[-1].get("content", "") or "").strip()
            parsed = self._parser.parse_answer(content) or ""

            if agent_id == "reverser":
                state["_reverser_output"] = content
                state["_reverser_parsed"] = parsed
                logger.info(
                    f"[ReverseMultiAgentEnv._agent_turn] reverser output: "
                    f"'{parsed[:50]}...' (len={len(parsed)})"
                )
            elif agent_id == "de_reverser":
                state["_de_reverser_output"] = content
                logger.info(
                    f"[ReverseMultiAgentEnv._agent_turn] de_reverser output: "
                    f"'{parsed[:50]}...' (len={len(parsed)})"
                )

        return step

    async def check_episode_done(self, state: vf.State) -> bool:
        """Episode ends after both agents have acted."""
        return state.get("current_turn", 0) >= 2


# =============================================================================
# Dataset
# =============================================================================


def create_dataset(
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL",
    dataset_split: str = "train",
) -> Dataset:
    """Load and prepare the dataset."""
    ds = load_dataset(dataset_name, split=dataset_split)

    # Map to standard format
    ds = ds.map(
        lambda x: {
            "question": x["prompt"],
            "answer": x["prompt"][::-1],  # For reference
            "info": {},
            "task": "reverse-multi-agent",
        }
    )
    ds = ds.remove_columns(["prompt"])

    return ds


# =============================================================================
# Entry Point
# =============================================================================


def load_environment(
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL",
    dataset_split: str = "train",
    **kwargs,
) -> ReverseMultiAgentEnv:
    """
    Load the multi-agent text reversal environment.

    Args:
        dataset_name: HuggingFace dataset to use
        dataset_split: Dataset split

    Returns:
        Configured multi-agent environment
    """
    dataset = create_dataset(dataset_name, dataset_split)

    # Create per-agent rubrics
    reverser_rubric = vf.Rubric()
    reverser_rubric.add_reward_func(reverser_reward, weight=1.0)

    de_reverser_rubric = vf.Rubric()
    de_reverser_rubric.add_reward_func(de_reverser_reward, weight=1.0)

    # Combine into multi-agent rubric
    rubric = vf.MultiAgentRubric(
        agent_rubrics={
            "reverser": reverser_rubric,
            "de_reverser": de_reverser_rubric,
        }
    )

    return ReverseMultiAgentEnv(
        dataset=dataset,
        rubric=rubric,
        **kwargs,
    )
