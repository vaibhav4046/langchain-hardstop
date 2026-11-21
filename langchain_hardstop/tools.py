"""LangChain tools for the five HARDSTOP MCP tools, with typed arguments."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_hardstop.client import HardstopCaller

Amount = float | str
AMOUNT_HELP = 'Dollar amount, for example 4 or "12.50". At most two decimals.'
TO_HELP = "Recipient address (0x...) or the name of a known recipient."


class _NoArgs(BaseModel):
    pass


class PaymentArgs(BaseModel):
    to: str = Field(description=TO_HELP)
    amount_usd: Amount = Field(description=AMOUNT_HELP)


class PayArgs(PaymentArgs):
    repeat: bool | None = Field(
        default=None,
        description="Set true only to pay the same recipient the same amount again within 10 minutes, on purpose.",
    )


class RequestMoreArgs(BaseModel):
    reason: str = Field(min_length=1, max_length=1000, description="Why you need more, in a sentence or two.")
    per_payment_usd: Amount | None = Field(default=None, description="Per-payment limit you are asking for.")
    per_day_usd: Amount | None = Field(default=None, description="Limit per spending window you are asking for.")
    recipient: str | None = Field(default=None, description="An address (0x...) you need the owner to approve.")


class HardstopTool(BaseTool):
    """Calls one HARDSTOP MCP tool. Returns the server's sentence to the model and its JSON as the artifact.

    Async only: the MCP session lives on one event loop, so use `ainvoke` (LangGraph does).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any = Field(exclude=True, description="A HardstopCaller, usually a connected HardstopClient.")
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    handle_tool_error: bool = True

    def _run(self, *args: Any, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError(f"{self.name} is async only; call it with `ainvoke`.")

    async def _arun(self, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        caller: HardstopCaller = self.client
        arguments = {k: v for k, v in kwargs.items() if v is not None and k != "run_manager"}
        result = await caller.call(self.name, arguments)
        if result.is_error:
            # Becomes a ToolMessage with status="error", so the model reads why instead of crashing the run.
            raise ToolException(result.text or f"{self.name} failed")
        return result.text, result.data


def _tool(name: str, description: str, schema: type[BaseModel], client: HardstopCaller) -> HardstopTool:
    return HardstopTool(name=name, description=description, args_schema=schema, client=client)


def hardstop_tools(client: HardstopCaller) -> list[HardstopTool]:
    """All five HARDSTOP agent tools, sharing one client. There are no owner tools, by design."""
    return [
        _tool(
            "hardstop_status",
            "Read the spending limits in force, what has been spent and is left in the current window, "
            "the vault balance, any waiting rule change, and the worst case over the next window. "
            "Changes nothing. Call this before planning any spending.",
            _NoArgs,
            client,
        ),
        _tool(
            "hardstop_preview_payment",
            "Ask the vault what it would answer right now to a payment, without sending anything. "
            "Returns would_pay and the reason in plain words.",
            PaymentArgs,
            client,
        ),
        _tool(
            "hardstop_pay",
            "Ask the vault to pay amount_usd to `to`. The vault's contract decides; it refuses anything "
            "outside the owner's limits. Returns paid, refused, reverted or not_sent with the reason.",
            PayArgs,
            client,
        ),
        _tool(
            "hardstop_recipients",
            "List the recipients known by name and whether the vault will pay each one. Changes nothing.",
            _NoArgs,
            client,
        ),
        _tool(
            "hardstop_request_more",
            "Draft a request to the vault owner for higher limits or a new recipient. Sends nothing and "
            "changes nothing; show the returned text to your user.",
            RequestMoreArgs,
            client,
        ),
    ]
