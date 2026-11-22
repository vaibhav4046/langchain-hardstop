# langchain-hardstop

LangChain tools and a LangGraph guard node for [HARDSTOP](https://github.com/vaibhav4046/hardstop),
a spending firewall for AI agents.

HARDSTOP puts an agent's spending limits in a smart contract the agent cannot talk its way past:
per-payment and per-window caps, approved recipients, and a cooling period before any limit can be
loosened. It ships an MCP server with five agent tools and no owner tools. This package makes that
server a first-class part of a LangChain / LangGraph agent:

- **`HardstopClient`**: starts the HARDSTOP MCP server once over stdio and keeps one session for the
  whole run (see [why one session](#design-notes)).
- **`hardstop_tools(client)`**: the five HARDSTOP tools as typed LangChain `BaseTool`s, with Pydantic
  argument schemas. Each returns the server's plain-language sentence to the model and its structured
  JSON as the tool message's `artifact`.
- **`create_hardstop_guard(tools, client)`**: a LangGraph node to use in place of a `ToolNode`. Every
  `hardstop_pay` call is previewed with the vault first; if the vault would refuse it, the pay tool
  never runs and the model gets the vault's reason as an error `ToolMessage`.
- **`create_guarded_agent(model, tools, client)`**: a minimal agent loop (`agent -> hardstop_guard -> agent`)
  built on `StateGraph(MessagesState)`.

> HARDSTOP runs on test networks only (Base Sepolia, Sepolia) and is an unaudited prototype.
> The demo mode used below is an in-memory network: nothing is real.

## Install

Not yet on PyPI. From GitHub:

```bash
pip install "langchain-hardstop @ git+https://github.com/vaibhav4046/langchain-hardstop"
```

Python 3.10+. The HARDSTOP server itself is Node: you need Node 20+ and `npx` on `PATH`
(the client runs `npx -y github:vaibhav4046/hardstop`).

## Quickstart (offline, no API keys)

```bash
git clone https://github.com/vaibhav4046/langchain-hardstop && cd langchain-hardstop
uv sync
uv run python examples/guarded_payment_agent.py
```

This starts the real HARDSTOP server in demo mode ($10 per payment, $20 a day, one approved
recipient called "Cloud hosting") and drives a LangGraph agent with a scripted chat model, so no
model API key is needed. Output from a run on 2026-11-21 (long lines cut):

```text
AI    -> hardstop_status({})
tool   : hardstop_status: HARDSTOP vault 0x2fe7...3871 on in-memory demo network (HARDSTOP_FAKE=1, nothing is real) ...
AI    -> hardstop_pay({'to': 'Cloud hosting', 'amount_usd': 4})
tool   : hardstop_pay: Paid: $4.00 to Cloud hosting. Inside every limit. $16.00 left today.
AI    -> hardstop_pay({'to': 'Cloud hosting', 'amount_usd': 15})
BLOCKED: hardstop_pay: Blocked by the HARDSTOP guard before sending: One payment may be at most $10.00. Nothing was sent.
AI    -> hardstop_request_more({'reason': 'The backup add-on costs $15, over the $10 per-payment limit.', 'per_payment_usd': 15})
tool   : hardstop_request_more: Request to the owner of HARDSTOP vault 0x2fe7...3871 on in-memory demo network ...
AI    : Paid $4 to Cloud hosting. The $15 backup add-on was blocked: one payment may be at most $10. ...

Vault after the run: spent $4.00 today.
```

The tool results, the refusal reason and the final balance come from the HARDSTOP server; only the
model's choices are scripted.

The example uses a scripted model. No hosted model has been run against this package; any chat model
that supports tool calling can be passed to `create_guarded_agent`, but that is untested here.

## Use it in your own graph

```python
from langgraph.graph import END, START, MessagesState, StateGraph
from langchain_hardstop import HardstopClient, create_hardstop_guard, hardstop_tools

async with HardstopClient.demo() as client:  # or HardstopClient.from_env(os.environ)
    tools = hardstop_tools(client)
    model = my_chat_model.bind_tools(tools)

    async def agent(state: MessagesState):
        return {"messages": [await model.ainvoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", create_hardstop_guard(tools, client))  # instead of ToolNode(tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END)
    graph.add_edge("tools", "agent")
    app = graph.compile()
    await app.ainvoke({"messages": [("user", "Pay Cloud hosting $4")]})
```

You can mix in your own tools: the guard runs any tool it is given and previews only `hardstop_pay`
(`pay_tool_name=` changes that). `check_payment(client, tool_call)` exposes the preview check on its
own if you want to build a different executor.

**A real vault.** `HardstopClient.from_env(os.environ)` passes only the `HARDSTOP_*` variables to the
server: `HARDSTOP_AGENT_KEY` (the agent's key, never the owner's), `HARDSTOP_VAULT` and
`HARDSTOP_CHAIN_ID`. See the [HARDSTOP MCP README](https://github.com/vaibhav4046/hardstop/tree/main/web/mcp)
for what each one means.

## Design notes

- **The contract decides.** Neither the guard nor the tools restate a limit. The guard asks the vault
  (`hardstop_preview_payment`) and acts on its `would_pay` answer; a payment that passes the preview is
  still checked by the contract when it is sent.
- **Fails closed.** If the preview errors or cannot be reached, the payment is blocked.
- **In order, one at a time.** Tool calls from one AI message run sequentially, so a second payment is
  previewed after the first has been paid (tested: three $9 payments against a $20 window give
  paid, paid, blocked).
- **Every tool call gets an answer.** A blocked call still gets a `ToolMessage` with its `tool_call_id`
  (status `"error"`), so the transcript stays valid for providers that require one result per call.
  The artifact is `{"blocked": True, "preview": {...}}`.
- **One session.** `langchain-mcp-adapters`' `MultiServerMCPClient.get_tools()` starts a new server
  process for each call (observed by the HARDSTOP project with v0.3.2). That loses the server's
  one-payment-at-a-time rule and its duplicate-payment check, and in demo mode gives each call a fresh
  vault. `HardstopClient` keeps one process for its `async with` block.
- **Why a node and not a separate pre-check?** LangGraph's `ToolNode` runs every tool call in the latest
  AI message, so a node placed *before* it cannot stop one call without rewriting the AI message. The
  guard is the executor instead. (`ToolNode` also accepts `awrap_tool_call`, which could host the same
  check; `check_payment` is exposed for that.)
- **Bad arguments do not crash the graph.** Arguments that fail a tool's schema, and any exception a
  tool raises, come back to the model as an error `ToolMessage`. A pay call is validated with the pay
  tool's schema before it is previewed, so the preview sees what would be sent.

## Tests

```bash
uv run pytest -m "not integration"   # 14 tests: fake in-process vault, scripted model, no network
uv run pytest -m integration          # 1 test: the real HARDSTOP MCP server in demo mode (needs npx)
uv run ruff check . && uv run mypy    # mypy --strict on the package
```

CI runs all of these, plus the example agent and a package build, on Python 3.10 and 3.13.

## Limitations

- **Async only.** The tools implement `_arun`; calling them synchronously raises `NotImplementedError`,
  because the MCP session belongs to one event loop.
- **The guard hides refusals from the chain.** HARDSTOP records a refused `hardstop_pay` on chain so
  the owner can see the attempt (at a small gas cost). A payment the guard blocks is never sent, so
  it does not appear in the vault's on-chain history; it appears only in your agent's transcript.
- **Preview and pay are two calls.** The vault can change between them (the owner pauses it, for
  example). The contract still refuses in that case; the guard just does not catch it first.
- **Start-up time.** `npx -y github:vaibhav4046/hardstop` fetches the server on first use; on the
  machine this was built on it took about 30 s before the first tool call.
- **Only the stdio transport** is wrapped. HARDSTOP's HTTP mode is not.
- Not published to PyPI yet.

## License

MIT. HARDSTOP itself is MIT, in its own repository.
