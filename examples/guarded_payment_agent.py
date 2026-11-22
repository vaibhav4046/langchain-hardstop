"""End-to-end demo: a LangGraph agent pays through HARDSTOP, and the guard stops a payment over the limit.

    python examples/guarded_payment_agent.py   # offline: scripted model, demo vault

This starts the real HARDSTOP MCP server in demo mode (`npx -y github:vaibhav4046/hardstop --demo`):
an in-memory vault with $10 per payment, $20 a day and one recipient, "Cloud hosting". Nothing is real.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_hardstop import HardstopClient, create_guarded_agent, hardstop_tools
from langchain_hardstop.testing import ScriptedChatModel

TASK = "Pay Cloud hosting $4 for this month's server, then $15 for the backup add-on."


def _call(name: str, call_id: str, **args: object) -> dict[str, object]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def scripted_model() -> ScriptedChatModel:
    """What a well-behaved model would do, fixed in advance so the demo runs offline."""
    return ScriptedChatModel(
        responses=[
            AIMessage("", tool_calls=[_call("hardstop_status", "c1")]),
            AIMessage("", tool_calls=[_call("hardstop_pay", "c2", to="Cloud hosting", amount_usd=4)]),
            AIMessage("", tool_calls=[_call("hardstop_pay", "c3", to="Cloud hosting", amount_usd=15)]),
            AIMessage(
                "",
                tool_calls=[
                    _call(
                        "hardstop_request_more",
                        "c4",
                        reason="The backup add-on costs $15, over the $10 per-payment limit.",
                        per_payment_usd=15,
                    )
                ],
            ),
            AIMessage(
                "Paid $4 to Cloud hosting. The $15 backup add-on was blocked: one payment may be at most $10. "
                "I drafted a request for the owner to raise the per-payment limit; it is above."
            ),
        ]
    )


def show(message: object) -> None:
    if isinstance(message, AIMessage):
        for call in message.tool_calls:
            print(f"AI    -> {call['name']}({call['args']})")
        if message.content:
            print(f"AI    : {message.text}")
    elif isinstance(message, ToolMessage):
        status = "BLOCKED" if isinstance(message.artifact, dict) and message.artifact.get("blocked") else "tool"
        first_line = str(message.content).splitlines()[0]
        print(f"{status:7}: {message.name}: {first_line}")


async def main() -> None:
    model = scripted_model()
    async with HardstopClient.demo() as client:
        agent = create_guarded_agent(model, hardstop_tools(client), client)
        result = await agent.ainvoke({"messages": [HumanMessage(TASK)]})
        for message in result["messages"]:
            show(message)
        status = await client.call("hardstop_status")
        print(f"\nVault after the run: spent ${status.data['spent_today_usd']} today.")


if __name__ == "__main__":
    asyncio.run(main())
