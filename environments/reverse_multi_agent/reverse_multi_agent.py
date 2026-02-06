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


SYSTEM_PROMPT = "Reverse the text character-by-character. Put your answer in <reversed_text> tags."


def lcs_ratio(x: str, y: str) -> float:
    """Return the longest common subsequence ratio of x and y."""
    return SequenceMatcher(None, x, y).ratio()


class ReverseMultiAgentEnv(vf.MultiAgentEnv):
    """
    Multi-agent environment for text reversal training.

    Two agents:
    - reverser: Takes original text, outputs reversal
    - de_reverser: Takes reverser's output, outputs its reversal
    """

    def __init__(
        self,
        dataset: Dataset,
        rubric: vf.Rubric,
        **kwargs: Any,
    ):
        self._parser = vf.XMLParser(["reversed_text"], answer_field="reversed_text")

        agents = {
            "reverser": AgentConfig(
                agent_id="reverser",
                system_prompt=SYSTEM_PROMPT,
                trainable=True,
                lora_id=0,
            ),
            "de_reverser": AgentConfig(
                agent_id="de_reverser",
                system_prompt=SYSTEM_PROMPT,
                trainable=True,
                lora_id=0,
            ),
        }

        super().__init__(
            agents=agents,
            turn_order="sequential",
            max_turns=2,
            dataset=dataset,
            rubric=rubric,
            parser=self._parser,
            **kwargs,
        )

    async def get_active_agents(self, state: vf.State) -> list[str]:
        current_turn = state.get("current_turn", 0)
        return ["reverser"] if current_turn == 0 else ["de_reverser"]

    async def get_initial_observation(self, agent_id: str, state: vf.State) -> str | None:
        if agent_id == "reverser":
            return state.get("original_text", "")
        return None

    async def get_agent_observation(
        self, agent_id: str, response: vf.Messages, state: vf.State
    ) -> str | None:
        if state.get("_last_agent") == "reverser" and agent_id == "de_reverser":
            return state.get("_reverser_parsed", "")
        return None

    async def setup_state(self, state: vf.State) -> vf.State:
        state["original_text"] = state.get("question", "")
        state["_reverser_parsed"] = ""
        state["_last_agent"] = None
        return await super().setup_state(state)

    async def _agent_turn(self, agent_id: str, state: vf.State) -> vf.TrajectoryStep:
        step = await super()._agent_turn(agent_id, state)
        state["_last_agent"] = agent_id

        completion = step.get("completion", [])
        if completion and agent_id == "reverser":
            content = (completion[-1].get("content", "") or "").strip()
            state["_reverser_parsed"] = self._parser.parse_answer(content) or ""

        return step

    async def check_episode_done(self, state: vf.State) -> bool:
        return state.get("current_turn", 0) >= 2


def create_reward_func(parser: vf.XMLParser, is_de_reverser: bool = False):
    """Create reward function matching reverse.py pattern exactly."""

    def lcs_reward_func(completion, answer, state, **kwargs) -> float:
        if is_de_reverser:
            # de_reverser's answer is reverser's output reversed
            answer = state.get("_reverser_parsed", "")[::-1]

        response = parser.parse_answer(completion) or ""
        return lcs_ratio(response, answer)

    return lcs_reward_func


def load_environment(
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL",
    dataset_split: str = "train",
    **kwargs,
) -> ReverseMultiAgentEnv:
    dataset = load_dataset(dataset_name, split=dataset_split).map(
        lambda x: {
            "question": x["prompt"],
            "answer": x["prompt"][::-1],
            "info": {},
            "task": "reverse-multi-agent",
        }
    )
    dataset = dataset.remove_columns(["prompt"])

    parser = vf.XMLParser(["reversed_text"], answer_field="reversed_text")

    reverser_rubric = vf.Rubric(
        funcs=[create_reward_func(parser, is_de_reverser=False)],
        weights=[1.0],
    )

    de_reverser_rubric = vf.Rubric(
        funcs=[create_reward_func(parser, is_de_reverser=True)],
        weights=[1.0],
    )

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
