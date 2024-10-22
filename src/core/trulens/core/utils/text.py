"""Utilities for user-facing text generation."""

import logging
import math
import sys

from trulens.core.utils import python as python_utils

logger = logging.getLogger(__name__)

if python_utils.safe_hasattr(sys.stdout, "reconfigure"):
    # Some stdout can't handle the below emojis (like terminal). This will skip over the emoji printing
    sys.stdout.reconfigure(errors="replace")

UNICODE_STOP = "🛑"
UNICODE_CHECK = "✅"
UNICODE_YIELD = "⚡"
UNICODE_HOURGLASS = "⏳"
UNICODE_CLOCK = "⏰"
UNICODE_SQUID = "🦑"
UNICODE_LOCK = "🔒"


class WithIdentString:  # want to be a Protocol but having metaclass conflicts
    """Mixin to indicate _ident_str is provided."""

    def _ident_str(self) -> str:
        """Get a string to identify this instance in some way without
        overburdening the output with details."""

        return f"{self.__class__.__name__} instance"


def make_retab(tab):
    def retab(s):
        lines = s.split("\n")
        return tab + f"\n{tab}".join(lines)

    return retab


def retab(s: str, tab: str = "\t"):
    return make_retab(tab)(s)


def _format_unit(
    unit: str,
    value: float,
    truncate: bool = True,
    precision: int = 2,
    pluralize: bool = True,
) -> str:
    if truncate:
        value = int(value)
    else:
        value = round(value, precision)
    if pluralize and value != 1:
        # This is a simple heuristic that works for most cases.
        return f"{value} {unit}s"
    return f"{value} {unit}"


def format_quantity(quantity: float, precision: int = 2) -> str:
    """Format a quantity into a human-readable string. This will use SI prefixes.
    Implementation details are largely copied from [millify](https://github.com/azaitsev/millify).

    Args:
        quantity (float): The quantity to format.
        precision (int, optional): The precision to use. Defaults to 2.

    Returns:
        str: The formatted quantity.
    """
    units = ["", "k", "M", "B", "T", "P", "E", "Z", "Y"]
    unit_idx = max(
        0,
        min(
            len(units) - 1,
            int(
                math.floor(
                    0 if quantity == 0 else math.log10(abs(quantity)) / 3
                )
            ),
        ),
    )
    result = "{:.{precision}f}".format(
        quantity / 10 ** (3 * unit_idx), precision=precision
    )
    return f"{result}{units[unit_idx]}"


def format_size(size: int) -> str:
    """Format a size (in bytes) into a human-readable string. This will use SI
    prefixes. Implementation details are largely copied from
    [millify](https://github.com/azaitsev/millify).

    Args:
        size: The quantity to format.

    Returns:
        str: The formatted quantity.
    """

    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    unit_idx = max(
        0,
        min(
            len(units) - 1,
            int(math.floor(0 if size == 0 else math.log10(abs(size)) / 3)),
        ),
    )
    result = f"{int(size / 10 ** (3 * unit_idx)):d}"

    return f"{result}{units[unit_idx]}"


def format_seconds(seconds: float, precision: int = 2) -> str:
    """Format seconds into human-readable time. This only goes up to days.

    Args:
        seconds (float): The number of seconds to format.
        precision (int, optional): The precision to use. Defaults to 2.

    Returns:
        str: The formatted time.
    """
    n_days = seconds // (60 * 60 * 24)
    n_hours = seconds // (60 * 60)
    n_minutes = seconds // 60
    if n_days:
        fmt_result = _format_unit("day", n_days, precision=precision)
        remaining_hours = seconds % (60 * 60 * 24) / (60 * 60)
        if remaining_hours >= 1:
            fmt_result += (
                f" {_format_unit('hour', remaining_hours, precision=precision)}"
            )
    elif n_hours:
        fmt_result = _format_unit("hour", n_hours, precision=precision)
        remaining_minutes = seconds % (60 * 60) / 60
        if remaining_minutes >= 1:
            fmt_result += f" {_format_unit('minute', remaining_minutes, precision=precision)}"
    elif n_minutes:
        fmt_result = _format_unit("minute", n_minutes, precision=precision)
        remaining_seconds = seconds % 60
        if remaining_seconds >= 1:
            fmt_result += f" {_format_unit('second', remaining_seconds, truncate=False, precision=precision)}"
    else:
        fmt_result = _format_unit(
            "second", seconds, truncate=False, precision=precision
        )
    return fmt_result
