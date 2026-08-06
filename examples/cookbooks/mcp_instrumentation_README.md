# MCP Tool Instrumentation with TruLens

This cookbook demonstrates how to trace and evaluate Model Context Protocol
(MCP) tool calls with TruLens. It uses the official MCP Python SDK, a local
deterministic server, and a LangGraph agent.

## What the example demonstrates

- Starting a local MCP server over the `stdio` transport.
- Discovering and invoking tools through the official MCP `ClientSession`.
- Exposing the discovered tools to a LangGraph agent.
- Recording MCP tool names, server names, arguments, outputs, errors, and
  execution times as TruLens spans.
- Evaluating tool selection and tool-call quality.
- Inspecting the recorded trace and evaluation results in the TruLens
  dashboard.

## Files

- `mcp_instrumentation.ipynb`: End-to-end instrumentation and evaluation
  walkthrough.
- `mcp_weather_server.py`: Local deterministic MCP server containing weather
  lookup and temperature conversion tools.

## Prerequisites

- Python supported by the TruLens repository.
- An OpenAI API key with available API quota.
- A notebook environment opened from either the repository root or the
  `examples/cookbooks` directory.

Set the API key in the environment before starting the notebook. Do not paste
the key into a notebook cell or commit it to Git.

PowerShell:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
jupyter notebook examples/cookbooks/mcp_instrumentation.ipynb
```

Bash:

```bash
export OPENAI_API_KEY="your-api-key"
jupyter notebook examples/cookbooks/mcp_instrumentation.ipynb
```

## How it works

The notebook starts `mcp_weather_server.py` as a subprocess using
`StdioServerParameters` and `stdio_client`. An official MCP `ClientSession`
initializes the connection, lists the available tools, and performs every tool
invocation.

Thin LangChain `StructuredTool` wrappers make the discovered MCP tools
available to LangGraph. The wrappers call an instrumented `call_mcp_tool`
function, which delegates execution to `ClientSession.call_tool`. The
instrumented function preserves the SDK result for tracing, including its
content types, text, and error status. The wrapper returns only the joined text
to LangGraph. This records the structured MCP response while keeping the
protocol integration on the official SDK.

For the example question, the agent should:

1. Call `get_weather` for Chicago.
2. Call `convert_temperature` to convert the Fahrenheit observation to
   Celsius.
3. Return a natural-language answer using the tool results.

## Run the cookbook

1. Open `mcp_instrumentation.ipynb`.
2. Run the dependency installation cell.
3. Restart the kernel if the notebook environment requests it.
4. Confirm that `OPENAI_API_KEY` is available to the kernel.
5. Run the remaining cells in order.
6. Open the dashboard and select the `mcp-weather-agent` record.

Expand the trace to inspect the MCP spans. Each span includes the tool name,
server name, JSON-serialized input arguments, structured SDK response content,
error status, and execution time. The record also displays the Tool Selection
and Tool Calling metric results after evaluation finishes.

## Local MCP validation

The MCP server can be started independently:

```bash
python examples/cookbooks/mcp_weather_server.py
```

The process communicates through standard input and output, so it normally
waits silently for an MCP client. Stop it with `Ctrl+C` when testing it
manually.

The server is deterministic and does not require network access. OpenAI access
is required only for the LangGraph model and the LLM-based evaluation metrics.

## Troubleshooting

### `OPENAI_API_KEY` is not set

Set the key in the terminal that starts Jupyter, then restart the notebook
kernel. Environment changes made in a different terminal do not automatically
propagate to an already-running kernel.

### HTTP 429 or `insufficient_quota`

The API key is recognized, but its project does not have usable quota. Check
the API project's billing and usage limits.

### The local server cannot be found

Start Jupyter from the repository root or `examples/cookbooks`. The notebook
checks both locations when resolving `mcp_weather_server.py`.

### The server remains running after an error

Run the notebook cleanup cell or restart the kernel. The main example closes
its MCP session in a `finally` block so normal model errors also trigger
cleanup.

## Security notes

- Never commit API keys, `.env` files, notebook outputs containing secrets, or
  local database files.
- Review MCP servers before running them. An MCP tool executes with the
  permissions of the Python process that starts it.
- This example intentionally uses a local server with deterministic,
  side-effect-free tools.
