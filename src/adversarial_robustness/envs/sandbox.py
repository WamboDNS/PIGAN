"""
PrimeIntellect Sandbox wrapper providing synchronous interface.

The PrimeIntellect SDK is async, but Ludic environments are synchronous.
This wrapper uses a background thread with a persistent event loop to bridge the gap.

Also provides MockSandboxWrapper for testing without PrimeIntellect infrastructure.
"""

from __future__ import annotations

import asyncio
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest
    PRIME_AVAILABLE = True
except ImportError:
    PRIME_AVAILABLE = False
    AsyncSandboxClient = None
    CreateSandboxRequest = None


@dataclass
class SandboxConfig:
    """Configuration for PrimeIntellect sandbox integration."""

    # Flag configuration
    flag_path: str = "/home/user/flag.txt"
    flag_format: str = "CTF{{{token}}}"  # Format string, {token} will be replaced
    flag_token_length: int = 8  # Random hex token length

    # Sandbox resources
    docker_image: str = "alpine:latest"
    cpu_cores: int = 1
    memory_gb: int = 1
    timeout_minutes: int = 30  # Sandbox lifetime
    network_access: bool = False  # Disable outbound for security

    # Available tools for Bob
    tools_enabled: List[str] = field(
        default_factory=lambda: ["read_file", "list_directory"]
    )
    enable_execute_command: bool = False  # Optional for harder scenarios


class SandboxWrapper:
    """
    Synchronous wrapper around the async PrimeIntellect sandbox SDK.

    Architecture:
    - Maintains a background thread with a persistent event loop
    - Sandbox client is created once and reused across episodes
    - Provides sync methods that block until async operations complete

    Usage:
        sandbox = SandboxWrapper(config)
        sandbox.start()
        try:
            flag = sandbox.setup_episode()  # Returns "CTF{...}"
            result = sandbox.execute_tool("read_file", {"path": "/home/user/flag.txt"})
        finally:
            sandbox.stop()
    """

    def __init__(self, config: SandboxConfig):
        self.config = config
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._client: Optional[AsyncSandboxClient] = None
        self._sandbox_id: Optional[str] = None
        self._current_flag: Optional[str] = None
        self._started = False

    def start(self) -> None:
        """Initialize background loop and sandbox client."""
        if self._started:
            return

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Initialize client in background loop
        self._run_async(self._async_init())
        self._started = True

    def _run_loop(self) -> None:
        """Run the event loop in background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_async(self, coro) -> Any:
        """Submit coroutine to background loop and block for result."""
        if not self._loop or not self._thread:
            raise RuntimeError("SandboxWrapper not started. Call start() first.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=300)  # 5 min timeout

    async def _async_init(self) -> None:
        """Initialize the async client."""
        self._client = AsyncSandboxClient()
        await self._client.__aenter__()

    async def _async_create_sandbox(self) -> str:
        """Create a new sandbox instance."""
        request = CreateSandboxRequest(
            name=f"injection-env-{secrets.token_hex(4)}",
            docker_image=self.config.docker_image,
            cpu_cores=self.config.cpu_cores,
            memory_gb=self.config.memory_gb,
            timeout_minutes=self.config.timeout_minutes,
            network_access=self.config.network_access,
        )
        sandbox = await self._client.create(request)
        await self._client.wait_for_creation(sandbox.id)
        return sandbox.id

    def _generate_flag(self) -> str:
        """Generate a new random flag using the configured format."""
        token = secrets.token_hex(self.config.flag_token_length // 2)
        return self.config.flag_format.format(token=token)

    async def _async_write_flag(self, flag: str) -> None:
        """Write the flag to the sandbox filesystem."""
        if not self._sandbox_id:
            raise RuntimeError("Sandbox not initialized")
        # Ensure parent directory exists and write flag
        parent_dir = "/".join(self.config.flag_path.rsplit("/", 1)[:-1]) or "/"
        await self._client.execute_command(
            self._sandbox_id, f'mkdir -p "{parent_dir}"'
        )
        cmd = f'echo "{flag}" > {self.config.flag_path}'
        await self._client.execute_command(self._sandbox_id, cmd)

    async def _async_setup_episode(self) -> str:
        """Setup sandbox for a new episode with fresh flag."""
        # Create sandbox if not exists (reuse existing sandbox)
        if not self._sandbox_id:
            self._sandbox_id = await self._async_create_sandbox()

        # Generate and write new flag
        self._current_flag = self._generate_flag()
        await self._async_write_flag(self._current_flag)

        return self._current_flag

    async def _async_reset_flag(self) -> str:
        """Replace the flag without recreating the sandbox."""
        if not self._sandbox_id:
            raise RuntimeError("Sandbox not initialized. Call setup_episode() first.")
        self._current_flag = self._generate_flag()
        await self._async_write_flag(self._current_flag)
        return self._current_flag

    async def _async_execute_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> str:
        """Execute a tool call in the sandbox."""
        if not self._sandbox_id:
            return "Error: Sandbox not initialized"

        if tool_name == "read_file":
            if "read_file" not in self.config.tools_enabled:
                return "Error: read_file tool is not enabled"
            path = arguments.get("path", "")
            result = await self._client.execute_command(
                self._sandbox_id, f'cat "{path}" 2>&1'
            )
            return result.stdout if hasattr(result, "stdout") else str(result)

        elif tool_name == "list_directory":
            if "list_directory" not in self.config.tools_enabled:
                return "Error: list_directory tool is not enabled"
            path = arguments.get("path", ".")
            result = await self._client.execute_command(
                self._sandbox_id, f'ls -la "{path}" 2>&1'
            )
            return result.stdout if hasattr(result, "stdout") else str(result)

        elif tool_name == "execute_command":
            if not self.config.enable_execute_command:
                return "Error: execute_command tool is not enabled"
            cmd = arguments.get("cmd", "")
            result = await self._client.execute_command(
                self._sandbox_id, cmd, timeout=10000  # 10 second timeout
            )
            return result.stdout if hasattr(result, "stdout") else str(result)

        else:
            return f"Error: Unknown tool '{tool_name}'"

    async def _async_cleanup(self) -> None:
        """Cleanup sandbox resources."""
        if self._sandbox_id and self._client:
            try:
                await self._client.delete(self._sandbox_id)
            except Exception:
                pass  # Best effort cleanup
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass

    # --- Public sync interface ---

    def setup_episode(self) -> str:
        """
        Setup sandbox for a new episode.

        Returns:
            The generated flag for this episode (e.g., "CTF{a8f3k2m9}")
        """
        return self._run_async(self._async_setup_episode())

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a tool call in the sandbox.

        Args:
            tool_name: Name of tool (read_file, list_directory, execute_command)
            arguments: Tool arguments dict

        Returns:
            Tool execution result as string
        """
        return self._run_async(self._async_execute_tool(tool_name, arguments))

    def reset_flag(self) -> str:
        """
        Replace the flag without recreating the sandbox.

        Use this between episodes for efficiency - the sandbox is reused
        and only the flag file contents are replaced.

        Returns:
            The new flag for this episode
        """
        return self._run_async(self._async_reset_flag())

    def destroy_sandbox(self) -> None:
        """
        Delete the current sandbox entirely.

        A new sandbox will be created on the next setup_episode() call.
        Use this sparingly - prefer reset_flag() between episodes.
        """
        if self._sandbox_id:

            async def _delete():
                await self._client.delete(self._sandbox_id)
                self._sandbox_id = None

            try:
                self._run_async(_delete())
            except Exception:
                self._sandbox_id = None

    def stop(self) -> None:
        """Cleanup resources and stop background thread."""
        if not self._started:
            return

        # Cleanup async resources
        try:
            self._run_async(self._async_cleanup())
        except Exception:
            pass

        # Stop the event loop
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        # Wait for thread to finish
        if self._thread:
            self._thread.join(timeout=5)

        self._started = False
        self._loop = None
        self._thread = None
        self._client = None
        self._sandbox_id = None

    @property
    def current_flag(self) -> Optional[str]:
        """Get the current episode's flag."""
        return self._current_flag

    @property
    def sandbox_id(self) -> Optional[str]:
        """Get the current sandbox ID."""
        return self._sandbox_id

    def __enter__(self) -> "SandboxWrapper":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


class MockSandboxWrapper:
    """
    Mock sandbox for testing without PrimeIntellect infrastructure.

    Simulates file system operations in memory. Useful for:
    - Local development and testing
    - CI/CD pipelines
    - Quick prototyping

    Usage:
        sandbox = MockSandboxWrapper(config)
        sandbox.start()
        flag = sandbox.setup_episode()  # Returns "CTF{...}"
        result = sandbox.execute_tool("read_file", {"path": "/home/user/flag.txt"})
    """

    def __init__(self, config: SandboxConfig):
        self.config = config
        self._started = False
        self._current_flag: Optional[str] = None
        self._filesystem: Dict[str, str] = {}  # path -> content

    def start(self) -> None:
        """Initialize the mock sandbox."""
        self._started = True
        self._filesystem = {
            "/home/user": "",  # Directory marker
            "/home": "",
            "/": "",
        }

    def _generate_flag(self) -> str:
        """Generate a new random flag."""
        token = secrets.token_hex(self.config.flag_token_length // 2)
        return self.config.flag_format.format(token=token)

    def setup_episode(self) -> str:
        """Setup for a new episode with a fresh flag."""
        if not self._started:
            self.start()

        self._current_flag = self._generate_flag()
        self._filesystem[self.config.flag_path] = self._current_flag
        return self._current_flag

    def reset_flag(self) -> str:
        """Replace the flag."""
        self._current_flag = self._generate_flag()
        self._filesystem[self.config.flag_path] = self._current_flag
        return self._current_flag

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call in the mock sandbox."""
        if tool_name == "read_file":
            if "read_file" not in self.config.tools_enabled:
                return "Error: read_file tool is not enabled"
            path = arguments.get("path", "")
            if path in self._filesystem:
                content = self._filesystem[path]
                if content:  # Non-empty = file
                    return content
                else:
                    return f"cat: {path}: Is a directory"
            return f"cat: {path}: No such file or directory"

        elif tool_name == "list_directory":
            if "list_directory" not in self.config.tools_enabled:
                return "Error: list_directory tool is not enabled"
            path = arguments.get("path", "/")
            if not path.endswith("/"):
                path_prefix = path + "/"
            else:
                path_prefix = path

            # Find entries in this directory
            entries = []
            for fpath in self._filesystem:
                if fpath.startswith(path_prefix) or fpath == path:
                    # Get the immediate child
                    rel = fpath[len(path_prefix):] if fpath != path else ""
                    if "/" not in rel and rel:
                        entries.append(rel)
            if not entries and path not in self._filesystem:
                return f"ls: cannot access '{path}': No such file or directory"
            return "\n".join(sorted(set(entries))) if entries else ""

        elif tool_name == "execute_command":
            if not self.config.enable_execute_command:
                return "Error: execute_command tool is not enabled"
            return "Mock: command execution not implemented"

        else:
            return f"Error: Unknown tool '{tool_name}'"

    def destroy_sandbox(self) -> None:
        """Reset the mock filesystem."""
        self._filesystem = {}
        self._current_flag = None

    def stop(self) -> None:
        """Stop the mock sandbox."""
        self._started = False
        self._filesystem = {}
        self._current_flag = None

    @property
    def current_flag(self) -> Optional[str]:
        """Get the current episode's flag."""
        return self._current_flag

    @property
    def sandbox_id(self) -> Optional[str]:
        """Mock sandbox ID."""
        return "mock-sandbox" if self._started else None

    def __enter__(self) -> "MockSandboxWrapper":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


def create_sandbox(config: SandboxConfig, mock: bool = False) -> "SandboxWrapper | MockSandboxWrapper":
    """
    Factory function to create appropriate sandbox wrapper.

    Args:
        config: Sandbox configuration
        mock: If True, use MockSandboxWrapper (no PrimeIntellect needed)
              If False, use real SandboxWrapper (requires prime_sandboxes)

    Returns:
        SandboxWrapper or MockSandboxWrapper instance
    """
    if mock:
        return MockSandboxWrapper(config)

    if not PRIME_AVAILABLE:
        raise ImportError(
            "prime_sandboxes not installed. Install with: "
            "pip install prime-sandboxes\n"
            "Or use mock=True for testing."
        )

    return SandboxWrapper(config)
