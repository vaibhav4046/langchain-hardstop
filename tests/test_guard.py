from __future__ import annotations

from typing import Any

import pytest
from fakes import RECIPIENT, FakeVault
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_hardstop import HardstopResult, create_guarded_agent, create_hardstop_guard, hardstop_tools
from langchain_hardstop.testing import ScriptedChatModel


def call(name: str, call_id: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


async def run_guard(vault: FakeVault, *calls: dict[str, Any]) -> list[ToolMessage]:
    guard = create_hardstop_guard(hardstop_tools(vault), vault)
    out = await guard({"messages": [HumanMessage("go"), AIMessage("", tool_calls=list(calls))]})
    messages = out["messages"]
    assert all(isinstance(m, ToolMessage) for m in messages)
    return messages  # type: ignore[return-value]


async def test_allowed_payment_is_previewed_then_paid() -> None:
    vault = FakeVault()
    [msg] = await run_guard(vault, call("hardstop_pay", "a", to=RECIPIENT, amount_usd=4))
    assert vault.names() == ["hardstop_preview_payment", "hardstop_pay"]
    assert msg.status == "success" and msg.tool_call_id == "a"
    assert msg.artifact == {"paid": True, "result": "paid"}


async def test_refused_payment_never_reaches_pay_tool() -> None:
    vault = FakeVault()
    [msg] = await run_guard(vault, call("hardstop_pay", "b", to=RECIPIENT, amount_usd=15))
    assert vault.names() == ["hardstop_preview_payment"]
    assert msg.status == "error" and msg.tool_call_id == "b"
    assert "at most $10.00" in str(msg.content) and "Nothing was sent" in str(msg.content)
    assert msg.artifact["blocked"] is True


async def test_unapproved_recipient_is_blocked() -> None:
    vault = FakeVault()
    [msg] = await run_guard(vault, call("hardstop_pay", "c", to="stranger", amount_usd=1))
    assert msg.status == "error" and "not approved" in str(msg.content)
    assert "hardstop_pay" not in vault.names()


async def test_preview_failure_fails_closed() -> None:
    vault = FakeVault()
    vault.fail_preview = True
    [msg] = await run_guard(vault, call("hardstop_pay", "d", to=RECIPIENT, amount_usd=1))
    assert msg.status == "error" and "ConnectionError" in str(msg.content)
    assert "hardstop_pay" not in vault.names()


async def test_calls_in_one_turn_run_in_order_and_see_earlier_payments() -> None:
    vault = FakeVault()
    msgs = await run_guard(
        vault,
        call("hardstop_pay", "1", to=RECIPIENT, amount_usd=9),
        call("hardstop_pay", "2", to=RECIPIENT, amount_usd=9),
        call("hardstop_pay", "3", to=RECIPIENT, amount_usd=9),  # 27 > 20 window
    )
    assert [m.status for m in msgs] == ["success", "success", "error"]
    assert [m.tool_call_id for m in msgs] == ["1", "2", "3"]
    assert vault.names().count("hardstop_pay") == 2


async def test_non_payment_tools_run_without_preview() -> None:
    vault = FakeVault()
    [msg] = await run_guard(vault, call("hardstop_status", "s"))
    assert vault.names() == ["hardstop_status"] and msg.status == "success"


async def test_unknown_tool_gets_error_message() -> None:
    vault = FakeVault()
    [msg] = await run_guard(vault, call("withdraw_everything", "x"))
    assert msg.status == "error" and "Unknown tool" in str(msg.content)
    assert vault.calls == []


async def test_server_error_becomes_error_tool_message() -> None:
    class Broken(FakeVault):
        async def call(self, name: str, arguments: Any = None) -> HardstopResult:
            return HardstopResult("vault unreachable", {}, is_error=True)

    vault = Broken()
    [msg] = await run_guard(vault, call("hardstop_status", "e"))
    assert msg.status == "error" and "vault unreachable" in str(msg.content)


async def test_no_ai_message_is_a_no_op() -> None:
    vault = FakeVault()
    guard = create_hardstop_guard(hardstop_tools(vault), vault)
    assert await guard({"messages": [HumanMessage("hi")]}) == {"messages": []}


def test_tools_are_async_only_and_have_no_owner_powers() -> None:
    tools = hardstop_tools(FakeVault())
    assert [t.name for t in tools] == [
        "hardstop_status",
        "hardstop_preview_payment",
        "hardstop_pay",
        "hardstop_recipients",
        "hardstop_request_more",
    ]
    with pytest.raises(NotImplementedError):
        tools[0].invoke({})


async def test_tool_drops_unset_optional_args() -> None:
    vault = FakeVault()
    pay = hardstop_tools(vault)[2]
    await pay.ainvoke({"to": RECIPIENT, "amount_usd": "2.50"})
    assert vault.calls[-1] == ("hardstop_pay", {"to": RECIPIENT, "amount_usd": "2.50"})


async def test_agent_loop_end_to_end_with_scripted_model() -> None:
    vault = FakeVault()
    model = ScriptedChatModel(
        responses=[
            AIMessage("", tool_calls=[call("hardstop_pay", "p1", to=RECIPIENT, amount_usd=4)]),
            AIMessage("", tool_calls=[call("hardstop_pay", "p2", to=RECIPIENT, amount_usd=15)]),
            AIMessage("Paid $4; $15 was blocked."),
        ]
    )
    agent = create_guarded_agent(model, hardstop_tools(vault), vault)
    result = await agent.ainvoke({"messages": [HumanMessage("pay")]})
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.status for m in tool_msgs] == ["success", "error"]
    assert result["messages"][-1].content == "Paid $4; $15 was blocked."
    assert vault.spent == 4
    assert vault.names() == ["hardstop_preview_payment", "hardstop_pay", "hardstop_preview_payment"]
