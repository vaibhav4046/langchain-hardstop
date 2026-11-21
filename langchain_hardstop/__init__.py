"""LangChain tools and a LangGraph guard node for HARDSTOP, a spending firewall for AI agents."""

from langchain_hardstop.agent import create_guarded_agent
from langchain_hardstop.client import HardstopCaller, HardstopClient, HardstopResult
from langchain_hardstop.guard import check_payment, create_hardstop_guard
from langchain_hardstop.tools import HardstopTool, hardstop_tools

__all__ = [
    "HardstopCaller",
    "HardstopClient",
    "HardstopResult",
    "HardstopTool",
    "check_payment",
    "create_guarded_agent",
    "create_hardstop_guard",
    "hardstop_tools",
]
__version__ = "0.1.0"
