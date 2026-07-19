# MCP Instrumentation

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open standard for
connecting LLM agents to external tools and data sources. TruLens provides first-class
support for instrumenting MCP tool calls via the `MCP` span type and a dedicated set of
semantic attributes.

## MCP span attributes

When you instrument an MCP tool call, TruLens captures the following attributes in the
`ai.observability.mcp.*` namespace:

| Attribute | Description | Type |
|-----------|-------------|------|
| `ai.observability.mcp.tool_name` | Name of the MCP tool being called | `str` |
| `ai.observability.mcp.tool_description` | Description of the MCP tool | `str` |
| `ai.observability.mcp.server_name` | Name of the MCP server providing the tool | `str` |
| `ai.observability.mcp.input_schema` | JSON schema of the tool's input parameters | `str` |
| `ai.observability.mcp.input_arguments` | Arguments passed to the tool | `str` |
| `ai.observability.mcp.output_content` | Content returned by the tool | `str` |
| `ai.observability.mcp.output_is_error` | Whether the tool call resulted in an error | `bool` |
| `ai.observability.mcp.execution_time_ms` | Time taken to execute the tool call (ms) | `float` |

## Instrumenting MCP tool calls

Use `@instrument` with `span_type=SpanAttributes.SpanType.MCP` to mark a method as an
MCP tool call. Map the relevant attributes using the `SpanAttributes.MCP` constants:

```python
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes


class MCPClient:
    @instrument(
        span_type=SpanAttributes.SpanType.MCP,
        attributes={
            SpanAttributes.MCP.TOOL_NAME: "tool_name",
            SpanAttributes.MCP.INPUT_ARGUMENTS: "arguments",
            SpanAttributes.MCP.OUTPUT_CONTENT: "return",
        },
    )
    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return its output."""
        ...
```

## Full working example

The example below shows a complete setup: an MCP client that calls a weather tool,
wrapped with TruLens instrumentation and evaluated for tool call quality.

```python
import json
from trulens.core import TruSession, Metric, Selector
from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.providers.openai import OpenAI

# --- App definition ---

class WeatherMCPClient:
    """Simulates an MCP client that calls a weather tool."""

    TOOLS = {
        "get_weather": {
            "description": "Get current weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        }
    }

    @instrument(
        span_type=SpanAttributes.SpanType.MCP,
        attributes={
            SpanAttributes.MCP.TOOL_NAME: "tool_name",
            SpanAttributes.MCP.TOOL_DESCRIPTION: lambda ret, exc, *args, **kwargs: (
                WeatherMCPClient.TOOLS.get(kwargs.get("tool_name", ""), {}).get(
                    "description", ""
                )
            ),
            SpanAttributes.MCP.INPUT_ARGUMENTS: lambda ret, exc, *args, **kwargs: (
                json.dumps(kwargs.get("arguments", {}))
            ),
            SpanAttributes.MCP.OUTPUT_CONTENT: "return",
            SpanAttributes.MCP.OUTPUT_IS_ERROR: lambda ret, exc, *args, **kwargs: (
                exc is not None
            ),
        },
    )
    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool."""
        if tool_name == "get_weather":
            city = arguments.get("city", "Unknown")
            units = arguments.get("units", "celsius")
            # Simulated tool response
            return json.dumps({
                "city": city,
                "temperature": 22 if units == "celsius" else 72,
                "units": units,
                "condition": "Partly cloudy",
            })
        raise ValueError(f"Unknown tool: {tool_name}")

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def answer(self, query: str) -> str:
        """Answer a user query by calling MCP tools."""
        result = self.call_tool("get_weather", {"city": "San Francisco", "units": "celsius"})
        data = json.loads(result)
        return (
            f"The weather in {data['city']} is {data['temperature']}°C "
            f"and {data['condition']}."
        )


# --- TruLens setup ---

session = TruSession()
provider = OpenAI()

metrics = [
    Metric(
        implementation=provider.tool_selection_with_cot_reasons,
        name="Tool Selection",
        selectors={"trace": Selector(trace_level=True)},
    ),
    Metric(
        implementation=provider.tool_calling_with_cot_reasons,
        name="Tool Calling",
        selectors={"trace": Selector(trace_level=True)},
    ),
    Metric(
        implementation=provider.tool_quality_with_cot_reasons,
        name="Tool Quality",
        selectors={"trace": Selector(trace_level=True)},
    ),
]

mcp_client = WeatherMCPClient()

tru_app = TruApp(
    mcp_client,
    app_name="WeatherMCPClient",
    app_version="v1",
    metrics=metrics,
)

# --- Run and evaluate ---

with tru_app as recording:
    response = mcp_client.answer("What's the weather in San Francisco?")

print(response)

session.get_leaderboard()
```

## Capturing execution time with lambda attributes

If your MCP client reports latency separately, you can capture it using a lambda:

```python
import time

class TimedMCPClient:
    @instrument(
        span_type=SpanAttributes.SpanType.MCP,
        attributes=lambda ret, exc, *args, **kwargs: {
            SpanAttributes.MCP.TOOL_NAME: kwargs.get("tool_name"),
            SpanAttributes.MCP.OUTPUT_CONTENT: ret,
            SpanAttributes.MCP.OUTPUT_IS_ERROR: exc is not None,
            # execution_time_ms captured from the tool response metadata
            SpanAttributes.MCP.EXECUTION_TIME_MS: (
                ret.get("latency_ms") if isinstance(ret, dict) else None
            ),
        },
    )
    def call_tool(self, tool_name: str, arguments: dict):
        ...
```

## Evaluating MCP tool calls

MCP spans are compatible with the agentic evaluators. To evaluate tool selection and
calling quality across MCP spans, use the Metric API with trace-level selectors:

```python
from trulens.core import Metric, Selector
from trulens.providers.openai import OpenAI

provider = OpenAI()

metrics = [
    Metric(
        implementation=provider.tool_selection_with_cot_reasons,
        name="Tool Selection",
        selectors={"trace": Selector(trace_level=True)},
    ),
    Metric(
        implementation=provider.tool_calling_with_cot_reasons,
        name="Tool Calling",
        selectors={"trace": Selector(trace_level=True)},
    ),
    Metric(
        implementation=provider.tool_quality_with_cot_reasons,
        name="Tool Quality",
        selectors={"trace": Selector(trace_level=True)},
    ),
]
```

See the [Metric migration guide](../evaluation/metric_migration.md) for the full API.

## Related

- [Instrumentation Overview](./index.md)
- [Metric Migration Guide](../evaluation/metric_migration.md)
- [Semantic Conventions](../../otel/semantic_conventions.md)
- [Feedback Selectors](../evaluation/feedback_selectors/index.md)
