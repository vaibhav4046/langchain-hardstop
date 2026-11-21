"""A single, long-lived MCP session with a HARDSTOP server."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Protocol

from mcp import Client, StdioServerParameters
from mcp.types import TextContent

DEMO_PACKAGE = "github:vaibhav4046/hardstop"


@dataclass(frozen=True)
class HardstopResult:
    """One tool answer: the server's plain-language text and its structured JSON."""

    text: str
    data: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False


class HardstopCaller(Protocol):
    """Anything that can call a HARDSTOP tool. `HardstopClient` is the real one; tests use fakes."""

    async def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> HardstopResult: ...


class HardstopClient:
    """Starts the HARDSTOP MCP server once over stdio and keeps one session open.

    One session matters: the server sends payments one at a time and remembers recent
    payments to refuse accidental duplicates, which only holds within one process.

    ```python
    async with HardstopClient.demo() as client:
        status = await client.call("hardstop_status")
    ```
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        # Resolve the executable so Windows finds `npx.cmd` and similar shims.
        resolved = shutil.which(command) or command
        self._params = StdioServerParameters(command=resolved, args=list(args or []), env=dict(env or {}), cwd=cwd)
        self._client: Client | None = None

    @classmethod
    def demo(cls) -> HardstopClient:
        """The published server in demo mode: an in-memory network and a freshly launched vault
        ($10 per payment, $20 a day, one approved recipient named "Cloud hosting"). No keys."""
        return cls("npx", ["-y", DEMO_PACKAGE, "--demo"])

    @classmethod
    def from_env(cls, env: Mapping[str, str], command: str = "npx", args: list[str] | None = None) -> HardstopClient:
        """A real vault. Pass `HARDSTOP_AGENT_KEY`, `HARDSTOP_VAULT` and `HARDSTOP_CHAIN_ID` (never the
        owner's key). They are passed explicitly because MCP stdio children get a minimal environment."""
        hardstop = {k: v for k, v in env.items() if k.startswith("HARDSTOP_")}
        return cls(command, args if args is not None else ["-y", DEMO_PACKAGE], env=hardstop)

    async def __aenter__(self) -> HardstopClient:
        client = Client(self._params)
        await client.__aenter__()
        self._client = client
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.__aexit__(exc_type, exc, tb)

    async def list_tool_names(self) -> list[str]:
        result = await self._session().list_tools()
        return [tool.name for tool in result.tools]

    async def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> HardstopResult:
        result = await self._session().call_tool(name, dict(arguments or {}))
        text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
        data = result.structured_content if isinstance(result.structured_content, dict) else {}
        return HardstopResult(text=text, data=data, is_error=bool(result.is_error))

    def _session(self) -> Client:
        if self._client is None:
            raise RuntimeError("HardstopClient is not connected; use `async with HardstopClient(...)`.")
        return self._client
