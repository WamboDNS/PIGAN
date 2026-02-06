"""
Multi-Agent Text Reversal Environment.

Two independent agents, both doing the same task: reverse text.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset
from verifiers.agents.agent import AgentConfig


SYSTEM_PROMPT = "Reverse the text character-by-character. Put your answer in <reversed_text> tags."


def lcs_ratio(x: str, y: str) -> float:
    return SequenceMatcher(None, x, y).ratio()


class ReverseMultiAgentEnv(vf.MultiAgentEnv):

    def __init__(self, dataset: Dataset, rubric: vf.Rubric, **kwargs: Any):
        self._parser = vf.XMLParser(["reversed_text"], answer_field="reversed_text")

        agents = {
            "agent_a": AgentConfig(
                agent_id="agent_a",
                system_prompt=SYSTEM_PROMPT,
                trainable=True,
                lora_id=0,
            ),
            "agent_b": AgentConfig(
                agent_id="agent_b",
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
        return ["agent_a"] if current_turn == 0 else ["agent_b"]

    async def get_initial_observation(self, agent_id: str, state: vf.State) -> str | None:
        # Both agents get the same input
        return state.get("question", "")

    async def setup_state(self, state: vf.State) -> vf.State:
        return await super().setup_state(state)

    async def check_episode_done(self, state: vf.State) -> bool:
        return state.get("current_turn", 0) >= 2


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

    # Same reward function as reverse.py
    def lcs_reward_func(completion, answer, **kwargs) -> float:
        response = parser.parse_answer(completion) or ""
        return lcs_ratio(response, answer)

    rubric = vf.Rubric(funcs=[lcs_reward_func], weights=[1.0])

    # Both agents use the same rubric
    multi_rubric = vf.MultiAgentRubric(
        agent_rubrics={
            "agent_a": rubric,
            "agent_b": rubric,
        }
    )

    return ReverseMultiAgentEnv(
        dataset=dataset,
        rubric=multi_rubric,
        **kwargs,
    )
