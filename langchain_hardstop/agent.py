"""A minimal LangGraph agent loop with the HARDSTOP guard in place of a ToolNode."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from langchain_hardstop.client import HardstopCaller
from langchain_hardstop.guard import create_hardstop_guard

SYSTEM_PROMPT = (
    "You pay for things from a HARDSTOP vault. Check hardstop_status before spending. "
    "If a payment is blocked or refused, do not retry it with a different amount to get around a limit; "
    "use hardstop_request_more and show the draft to the user."
)


def create_guarded_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    client: HardstopCaller,
    system_prompt: str = SYSTEM_PROMPT,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """agent -> hardstop_guard -> agent ... until the model answers without tool calls."""
    bound = model.bind_tools(list(tools))

    async def agent(state: MessagesState) -> dict[str, list[Any]]:
        reply = await bound.ainvoke([SystemMessage(system_prompt), *state["messages"]])
        return {"messages": [reply]}

    guard = create_hardstop_guard(tools, client)

    async def hardstop_guard(state: MessagesState) -> dict[str, list[Any]]:
        return await guard(state)

    def route(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "hardstop_guard" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("hardstop_guard", hardstop_guard)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["hardstop_guard", END])
    graph.add_edge("hardstop_guard", "agent")
    return graph.compile()
