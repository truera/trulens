"""Small local MCP server used by the MCP instrumentation cookbook."""

from mcp.server import fastmcp

server = fastmcp.FastMCP("local-weather")


@server.tool()
def get_weather(city: str) -> str:
    """Return deterministic weather for a supported city."""
    weather = {
        "chicago": "Chicago is 72 F with light wind.",
        "london": "London is 61 F with scattered showers.",
        "tokyo": "Tokyo is 79 F and partly cloudy.",
    }
    return weather.get(
        city.casefold(),
        f"No weather observation is available for {city}.",
    )


@server.tool()
def convert_temperature(value: float, to_unit: str) -> str:
    """Convert a temperature to Celsius or Fahrenheit."""
    unit = to_unit.casefold()
    if unit in {"c", "celsius"}:
        converted = (value - 32) * 5 / 9
        return f"{value:g} F is {converted:.1f} C."
    if unit in {"f", "fahrenheit"}:
        converted = value * 9 / 5 + 32
        return f"{value:g} C is {converted:.1f} F."
    raise ValueError("to_unit must be Celsius/C or Fahrenheit/F")


if __name__ == "__main__":
    server.run(transport="stdio")
