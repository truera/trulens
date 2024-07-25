"""Serializable selector-related classes."""

from __future__ import annotations

import logging
from typing import Tuple, TypeVar

from trulens.utils import serial

T = TypeVar("T")

logger = logging.getLogger(__name__)


class Select:
    """
    Utilities for creating selectors using Lens and aliases/shortcuts.
    """

    # TODEP
    Query = serial.Lens
    """Selector type."""

    Tru: serial.Lens = Query()
    """Selector for the tru wrapper (TruLlama, TruChain, etc.)."""

    Record: Query = Query().__record__
    """Selector for the record."""

    App: Query = Query().__app__
    """Selector for the app."""

    RecordInput: Query = Record.main_input
    """Selector for the main app input."""

    RecordOutput: Query = Record.main_output
    """Selector for the main app output."""

    RecordCalls: Query = Record.app  # type: ignore
    """Selector for the calls made by the wrapped app.

    Layed out by path into components.
    """

    RecordCall: Query = Record.calls[-1]
    """Selector for the first called method (last to return)."""

    RecordArgs: Query = RecordCall.args
    """Selector for the whole set of inputs/arguments to the first called / last method call."""

    RecordRets: Query = RecordCall.rets
    """Selector for the whole output of the first called / last returned method call."""

    @staticmethod
    def path_and_method(select: Select.Query) -> Tuple[Select.Query, str]:
        """
        If `select` names in method as the last attribute, extract the method name
        and the selector without the final method name.
        """

        if len(select.path) == 0:
            raise ValueError(
                "Given selector is empty so does not name a method."
            )

        firsts = select.path[:-1]
        last = select.path[-1]

        if not isinstance(last, serial.StepItemOrAttribute):
            raise ValueError(
                "Last part of selector is not an attribute so does not name a method."
            )

        method_name = last.get_item_or_attribute()
        path = Select.Query(path=firsts)

        return path, method_name

    @staticmethod
    def dequalify(select: Select.Query) -> Select.Query:
        """If the given selector qualifies record or app, remove that qualification."""

        if len(select.path) == 0:
            return select

        if (
            select.path[0] == Select.Record.path[0]
            or select.path[0] == Select.App.path[0]
        ):
            return Select.Query(path=select.path[1:])

        return select

    @staticmethod
    def for_record(query: Select.Query) -> Query:
        return Select.Query(path=Select.Record.path + query.path)

    @staticmethod
    def for_app(query: Select.Query) -> Query:
        return Select.Query(path=Select.App.path + query.path)

    @staticmethod
    def render_for_dashboard(query: Select.Query) -> str:
        """Render the given query for use in dashboard to help user specify feedback functions."""

        if len(query) == 0:
            return "Select.Query()"

        ret = ""
        rest = None

        if query.path[0:2] == Select.RecordInput.path:
            ret = "Select.RecordInput"
            rest = query.path[2:]
        elif query.path[0:2] == Select.RecordOutput.path:
            ret = "Select.RecordOutput"
            rest = query.path[2:]

        elif query.path[0:4] == Select.RecordArgs.path:
            ret = "Select.RecordArgs"
            rest = query.path[4:]
        elif query.path[0:4] == Select.RecordRets.path:
            ret = "Select.RecordRets"
            rest = query.path[4:]

        elif query.path[0:2] == Select.RecordCalls.path:
            ret = "Select.RecordCalls"
            rest = query.path[2:]

        elif query.path[0:3] == Select.RecordCall.path:
            ret = "Select.RecordCall"
            rest = query.path[3:]

        elif query.path[0] == Select.Record.path[0]:
            ret = "Select.Record"
            rest = query.path[1:]
        elif query.path[0] == Select.App.path[0]:
            ret = "Select.App"
            rest = query.path[1:]
        else:
            rest = query.path

        for step in rest:
            ret += repr(step)

        return f"{ret}"
