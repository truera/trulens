"""Debugging utilities."""

import asyncio
import io
import threading
from typing import Optional

from trulens.core.utils import python as python_utils
from trulens.core.utils import text as text_utils


def iprint(*args, tabs=0, **kwargs):
    """Print with thread and task info,"""

    thread = threading.current_thread()
    thread_ident = thread.name

    try:
        task = asyncio.current_task()
        task_ident = task.get_name()
    except RuntimeError:
        task_ident = "no running loop"

    textbuffer = io.StringIO()

    print(*args, **kwargs, file=textbuffer, end="")

    text = textbuffer.getvalue()

    tabbed = text_utils.retab(
        text, tab=f"[{thread_ident}][{task_ident}]" + ("    " * tabs) + " "
    )

    print(tabbed)


def print_context(msg: Optional[str] = None, tabs=0):
    """Print the status of trulens context variables."""

    from trulens.core import instruments as core_instruments
    from trulens.core.feedback import endpoint as core_endpoint

    if msg is None:
        msg = python_utils.code_line(
            python_utils.caller_frameinfo(offset=1), show_source=True
        )

    contextvars = [
        core_instruments.WithInstrumentCallbacks._context_contexts,
        core_instruments.WithInstrumentCallbacks._stack_contexts,
        core_endpoint.Endpoint._context_endpoints,
    ]

    iprint(msg, tabs=tabs)

    for contextvar in contextvars:
        iprint(f"{contextvar.name} [{len(contextvar.get())}]", tabs=tabs + 1)
