"""A LangGraph node that runs an agent's tool calls, previewing every payment with the vault first."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool

from langchain_hardstop.client import HardstopCaller

PAY_TOOL = "hardstop_pay"
PREVIEW_TOOL = "hardstop_preview_payment"

GuardNode = Callable[[Mapping[str, Any]], Awaitable[dict[str, list[AnyMessage]]]]


def _blocked(call: ToolCall, sentence: str, preview: Mapping[str, Any]) -> ToolMessage:
    return ToolMessage(
        content=f"Blocked by the HARDSTOP guard before sending: {sentence} Nothing was sent.",
        tool_call_id=call["id"] or "",
        name=call["name"],
        status="error",
        artifact={"blocked": True, "preview": dict(preview)},
    )


async def check_payment(client: HardstopCaller, call: ToolCall) -> ToolMessage | None:
    """Preview a pay call with the vault. Returns a blocking ToolMessage, or None if the vault would pay.

    Fails closed: if the preview cannot be read, the payment is blocked.
    """
    args = call["args"]
    try:
        result = await client.call(PREVIEW_TOOL, {"to": args.get("to"), "amount_usd": args.get("amount_usd")})
    except Exception as error:  # noqa: BLE001 - any failure to preview means do not pay
        return _blocked(call, f"the payment could not be previewed ({type(error).__name__}).", {})
    if result.is_error:
        return _blocked(call, result.text, result.data)
    if result.data.get("would_pay") is True:
        return None
    sentence = str(result.data.get("explanation") or "The vault would refuse this payment.")
    return _blocked(call, sentence, result.data)


def create_hardstop_guard(
    tools: Sequence[BaseTool],
    client: HardstopCaller,
    *,
    pay_tool_name: str = PAY_TOOL,
    messages_key: str = "messages",
) -> GuardNode:
    """Build a node to use in place of a `ToolNode`.

    For each tool call in the latest AI message it runs the tool, except that a call to
    `pay_tool_name` is first previewed with `hardstop_preview_payment`; if the vault would refuse,
    the pay tool never runs and the model gets a ToolMessage (status "error") with the vault's reason.
    The contract still decides every payment that is sent; the guard only stops ones it already knows
    would be refused.
    """
    by_name = {tool.name: tool for tool in tools}

    async def hardstop_guard(state: Mapping[str, Any]) -> dict[str, list[AnyMessage]]:
        messages: Sequence[AnyMessage] = state[messages_key]
        ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if ai is None:
            return {messages_key: []}
        out: list[AnyMessage] = []
        # One at a time, in order: a payment can change what the next preview answers.
        for call in ai.tool_calls:
            if call["name"] == pay_tool_name:
                blocked = await check_payment(client, call)
                if blocked is not None:
                    out.append(blocked)
                    continue
            tool = by_name.get(call["name"])
            if tool is None:
                out.append(
                    ToolMessage(
                        content=f"Unknown tool {call['name']!r}. Available: {', '.join(sorted(by_name))}.",
                        tool_call_id=call["id"] or "",
                        name=call["name"],
                        status="error",
                    )
                )
                continue
            out.append(await tool.ainvoke(call))
        return {messages_key: out}

    return hardstop_guard
