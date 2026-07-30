"""Integration tests for the MCP instrumentation cookbook server."""

from pathlib import Path
import sys

import pytest

mcp = pytest.importorskip("mcp")

from mcp import ClientSession  # noqa: E402
from mcp import StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


@pytest.mark.asyncio
async def test_mcp_weather_server_tools() -> None:
    """Discover and invoke both tools through the official MCP SDK."""
    repository_root = Path(__file__).parents[2]
    server_path = repository_root / "examples/cookbooks/mcp_weather_server.py"
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    async with stdio_client(server_parameters) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools

            assert {tool.name for tool in tools} == {
                "convert_temperature",
                "get_weather",
            }

            weather = await session.call_tool(
                "get_weather", {"city": "Chicago"}
            )
            conversion = await session.call_tool(
                "convert_temperature",
                {"value": 72, "to_unit": "Celsius"},
            )

    assert weather.isError is False
    assert weather.content[0].text == "Chicago is 72 F with light wind."
    assert conversion.isError is False
    assert conversion.content[0].text == "72 F is 22.2 C."
