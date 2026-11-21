# Draft: "Use with LangChain" section for the HARDSTOP README

Not yet added to vaibhav4046/hardstop: that repository is frozen for the Fe Builders Challenge until
after 25 Nov 2026. Proposed text, for `web/mcp/README.md` (after the Install table) or the main README:

---

## Use with LangChain / LangGraph

[`langchain-hardstop`](https://github.com/vaibhav4046/langchain-hardstop) wraps this server for
LangChain and LangGraph: the five agent tools as typed `BaseTool`s over one long-lived MCP session,
and a guard node that previews every `hardstop_pay` with the vault and stops a payment the vault
would refuse before it is sent. The vault still decides every payment that is sent.

```python
from langchain_hardstop import HardstopClient, create_guarded_agent, hardstop_tools

async with HardstopClient.demo() as client:  # npx -y github:vaibhav4046/hardstop --demo
    agent = create_guarded_agent(model, hardstop_tools(client), client)
    await agent.ainvoke({"messages": [("user", "Pay Cloud hosting $15")]})
    # -> "Blocked by the HARDSTOP guard before sending: One payment may be at most $10.00."
```

Tested in that repository's CI against this server's demo mode, with a scripted model (no API key).
