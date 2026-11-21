"""Against the real HARDSTOP MCP server in demo mode (in-memory vault, no keys). Needs Node 20+ and npx."""

from __future__ import annotations

import shutil

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_hardstop import HardstopClient, create_guarded_agent, hardstop_tools
from langchain_hardstop.testing import ScriptedChatModel

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("npx") is None, reason="npx not found"),
]


def call(name: str, call_id: str, **args: object) -> dict[str, object]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


async def test_guarded_agent_against_real_demo_vault() -> None:
    async with HardstopClient.demo() as client:
        names = await client.list_tool_names()
        assert set(names) == {
            "hardstop_status",
            "hardstop_preview_payment",
            "hardstop_pay",
            "hardstop_recipients",
            "hardstop_request_more",
        }
        model = ScriptedChatModel(
            responses=[
                AIMessage("", tool_calls=[call("hardstop_pay", "p1", to="Cloud hosting", amount_usd=4)]),
                AIMessage("", tool_calls=[call("hardstop_pay", "p2", to="Cloud hosting", amount_usd=15)]),
                AIMessage("done"),
            ]
        )
        agent = create_guarded_agent(model, hardstop_tools(client), client)
        result = await agent.ainvoke({"messages": [HumanMessage("pay")]})
        paid, blocked = [m for m in result["messages"] if isinstance(m, ToolMessage)]

        assert paid.status == "success" and paid.artifact["result"] == "paid"
        assert blocked.status == "error" and blocked.artifact["blocked"] is True
        assert blocked.artifact["preview"]["reason"] == "OverPaymentLimit"

        status = await client.call("hardstop_status")
        assert status.data["spent_today_usd"] == "4.00"  # the $15 was never sent
