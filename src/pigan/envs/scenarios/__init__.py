"""Scenario definitions for prompt injection environment."""

from .presets import (
    EASY_SCENARIOS,
    MEDIUM_SCENARIOS,
    HARD_SCENARIOS,
    ALL_SCENARIOS,
    get_scenarios_by_difficulty,
)

__all__ = [
    "EASY_SCENARIOS",
    "MEDIUM_SCENARIOS",
    "HARD_SCENARIOS",
    "ALL_SCENARIOS",
    "get_scenarios_by_difficulty",
]
