"""
Alice and Bob simulators using local vLLM instances.

These are non-trained entities that provide the conversation substrate:
- Alice: Simulated user
- Bob: Simulated assistant with tool-use access to sandbox
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ludic.inference.vllm_client import VLLMChatClient
from ludic.inference.chat_template import ChatTemplate
from ludic.inference.request import SamplingParams
from ludic.types import ChatResponse

from .sandbox import SandboxWrapper
from .scenario import BobToolConfig


@dataclass
class SimulatorConfig:
    """Configuration for simulators."""

    model: str
    temperature: float = 0.7
    max_tokens: int = 512


class AliceSimulator:
    """
    Simulates Alice (the user) using a local vLLM instance.

    Alice has natural conversations and may reveal information
    if successfully manipulated by injected messages.
    """

    def __init__(
        self,
        client: VLLMChatClient,
        chat_template: ChatTemplate,
        config: SimulatorConfig,
    ):
        self.client = client
        self.chat_template = chat_template
        self.config = config

    async def generate_response(
        self,
        conversation_history: List[Dict[str, str]],
        system_prompt: str,
    ) -> str:
        """
        Generate Alice's response to the conversation.

        Args:
            conversation_history: List of {"role": "user"|"assistant", "content": str}
            system_prompt: Alice's system prompt defining her persona

        Returns:
            Alice's response text
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)

        # Apply chat template
        template_result = self.chat_template.apply(
            messages, add_generation_prompt=True
        )

        # Create completion request
        from ludic.inference.request import TokenCompletionRequest, ReturnSpec

        request = TokenCompletionRequest(
            model=self.config.model,
            prompt_token_ids=template_result.prompt_token_ids,
            sampling=SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ),
            return_=ReturnSpec(return_chosen_logprobs=False),
        )

        response, _ = await self.client.complete_tokens(request)

        return response.text or ""

    def generate_response_sync(
        self,
        conversation_history: List[Dict[str, str]],
        system_prompt: str,
    ) -> str:
        """Synchronous wrapper for generate_response."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # We're in an async context, need to run in executor
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.generate_response(conversation_history, system_prompt),
                )
                return future.result()
        else:
            return asyncio.run(
                self.generate_response(conversation_history, system_prompt)
            )


class BobSimulator:
    """
    Simulates Bob (the assistant with tool-use) using a local vLLM instance.

    Bob can use tools to access the sandbox filesystem. Tool calls are
    parsed from the model output and executed against the sandbox.

    Tool-use format (Hermes-style):
        <tool_call>{"name": "read_file", "arguments": {"path": "/path"}}</tool_call>
    """

    # Regex to find tool calls in output
    TOOL_CALL_PATTERN = re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
    )

    def __init__(
        self,
        client: VLLMChatClient,
        chat_template: ChatTemplate,
        sandbox: SandboxWrapper,
        config: SimulatorConfig,
        tool_config: Optional[BobToolConfig] = None,
    ):
        self.client = client
        self.chat_template = chat_template
        self.sandbox = sandbox
        self.config = config
        self.tool_config = tool_config or BobToolConfig()

    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from model output.

        Returns:
            List of {"name": str, "arguments": dict} dicts
        """
        tool_calls = []
        for match in self.TOOL_CALL_PATTERN.finditer(text):
            try:
                call_json = json.loads(match.group(1))
                if "name" in call_json:
                    tool_calls.append(call_json)
            except json.JSONDecodeError:
                continue
        return tool_calls

    def _format_tool_results(
        self, tool_calls: List[Dict[str, Any]], results: List[str]
    ) -> str:
        """Format tool results for injection into conversation."""
        parts = []
        for call, result in zip(tool_calls, results):
            parts.append(
                f"<tool_result name=\"{call['name']}\">\n{result}\n</tool_result>"
            )
        return "\n".join(parts)

    async def _generate_once(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """Single generation call."""
        template_result = self.chat_template.apply(
            messages, add_generation_prompt=True
        )

        from ludic.inference.request import TokenCompletionRequest, ReturnSpec

        request = TokenCompletionRequest(
            model=self.config.model,
            prompt_token_ids=template_result.prompt_token_ids,
            sampling=SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ),
            return_=ReturnSpec(return_chosen_logprobs=False),
        )

        response, _ = await self.client.complete_tokens(request)
        return response.text or ""

    async def generate_response(
        self,
        conversation_history: List[Dict[str, str]],
        system_prompt: str,
    ) -> str:
        """
        Generate Bob's response, handling tool-use iterations.

        The model may output tool calls which are executed against the sandbox.
        Results are fed back and the model continues until it produces a
        final response without tool calls.

        Args:
            conversation_history: List of {"role": "user"|"assistant", "content": str}
            system_prompt: Bob's system prompt (should include tool definitions)

        Returns:
            Bob's final response text (may include tool outputs)
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)

        all_content_parts = []

        for iteration in range(self.tool_config.max_tool_iterations):
            response_text = await self._generate_once(messages)

            # Check for tool calls
            tool_calls = self._parse_tool_calls(response_text)

            if not tool_calls:
                # No tool calls - this is the final response
                # Combine any previous content with this response
                if all_content_parts:
                    # Remove tool call tags from response for cleaner output
                    clean_response = self.TOOL_CALL_PATTERN.sub("", response_text).strip()
                    all_content_parts.append(clean_response)
                    return "\n".join(all_content_parts)
                return response_text

            # Execute tool calls against sandbox
            results = []
            for call in tool_calls:
                tool_name = call.get("name", "")
                arguments = call.get("arguments", {})
                result = self.sandbox.execute_tool(tool_name, arguments)
                results.append(result)

            # Extract content before tool calls (if any)
            content_before_tools = self.TOOL_CALL_PATTERN.split(response_text)[0].strip()
            if content_before_tools:
                all_content_parts.append(content_before_tools)

            # Add assistant message with tool calls
            messages.append({"role": "assistant", "content": response_text})

            # Add tool results as a system/user message
            tool_results_text = self._format_tool_results(tool_calls, results)
            messages.append({"role": "user", "content": f"Tool results:\n{tool_results_text}"})

        # Exhausted iterations - force final response by removing tools from prompt
        # Generate one more time with a prompt to conclude
        messages.append({
            "role": "user",
            "content": "Please provide your final response based on the information gathered."
        })

        final_response = await self._generate_once(messages)

        if all_content_parts:
            all_content_parts.append(final_response)
            return "\n".join(all_content_parts)
        return final_response

    def generate_response_sync(
        self,
        conversation_history: List[Dict[str, str]],
        system_prompt: str,
    ) -> str:
        """Synchronous wrapper for generate_response."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # We're in an async context, need to run in executor
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.generate_response(conversation_history, system_prompt),
                )
                return future.result()
        else:
            return asyncio.run(
                self.generate_response(conversation_history, system_prompt)
            )


def create_simulators(
    client: VLLMChatClient,
    chat_template: ChatTemplate,
    sandbox: SandboxWrapper,
    model: str,
    alice_temperature: float = 0.7,
    bob_temperature: float = 0.7,
    max_tokens: int = 512,
) -> Tuple[AliceSimulator, BobSimulator]:
    """
    Create Alice and Bob simulators with shared client.

    Args:
        client: vLLM client for the simulator instance
        chat_template: Chat template for the model
        sandbox: Sandbox wrapper for Bob's tool execution
        model: Model name
        alice_temperature: Sampling temperature for Alice
        bob_temperature: Sampling temperature for Bob
        max_tokens: Max tokens per response

    Returns:
        Tuple of (AliceSimulator, BobSimulator)
    """
    alice_config = SimulatorConfig(
        model=model,
        temperature=alice_temperature,
        max_tokens=max_tokens,
    )
    bob_config = SimulatorConfig(
        model=model,
        temperature=bob_temperature,
        max_tokens=max_tokens,
    )

    alice = AliceSimulator(client, chat_template, alice_config)
    bob = BobSimulator(client, chat_template, sandbox, bob_config)

    return alice, bob
