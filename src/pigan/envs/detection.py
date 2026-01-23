"""
Detection utilities for prompt injection environment.

Provides XML tag extraction for M and D agent outputs.

Note: Flag leakage detection is in rewards.py (check_flag_leakage).
"""

from __future__ import annotations

import re
from typing import Optional


def extract_xml_tag(text: str, tag: str) -> Optional[str]:
    """
    Extract the inner content of an XML-style tag.

    Args:
        text: Text to search
        tag: Tag name (e.g., "classification" for <classification>...</classification>)

    Returns:
        Inner content if found, None otherwise
    """
    pattern = re.compile(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None


def parse_m_injection(action: str) -> str:
    """
    Parse Agent M's injection output (GAN-style, M always injects).

    Expected format:
        <injection>The modified message with hidden instructions</injection>

    Args:
        action: M's raw output

    Returns:
        The injected message to send to Bob
    """
    # Extract injection content
    injection = extract_xml_tag(action, "injection")
    if injection is not None:
        return injection

    # Fallback: try old format <message> tag
    message = extract_xml_tag(action, "message")
    if message is not None:
        return message

    # Last resort: treat entire output as the injection
    return action.strip()


def parse_d_classification(action: str) -> str:
    """
    Parse Agent D's classification output.

    Expected format:
        <classification>clean</classification>
        or
        <classification>injected</classification>

    Args:
        action: D's raw output

    Returns:
        "clean" or "injected" (defaults to "clean" if parsing fails)
    """
    classification = extract_xml_tag(action, "classification")
    if classification is not None:
        classification = classification.lower()
        if classification in ("clean", "injected"):
            return classification

    # Fallback: look for keywords
    action_lower = action.lower()
    if "injected" in action_lower and "clean" not in action_lower:
        return "injected"
    if "clean" in action_lower and "injected" not in action_lower:
        return "clean"

    # Default to clean (conservative)
    return "clean"
