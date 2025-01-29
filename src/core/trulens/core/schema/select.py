"""Serializable selector-related classes."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, TypeVar

from trulens.core._utils.pycompat import TypeAlias
from trulens.core.utils import deprecation as deprecation_utils
from trulens.core.utils import serial as serial_utils

T = TypeVar("T")

logger = logging.getLogger(__name__)


class Select:
    """
    Utilities for creating selectors using Lens and aliases/shortcuts.
    """

    Lens: TypeAlias = serial_utils.Lens

    Query: TypeAlias = Lens  # Deprecate this alias in the future.

    Tru: Lens = Lens()
    """Selector for the tru wrapper (TruLlama, TruChain, etc.)."""

    Record: Lens = Lens().__record__
    """Selector for the record."""

    App: Lens = Lens().__app__
    """Selector for the app."""

    RecordInput: Lens = Record.main_input
    """Selector for the main app input."""

    RecordOutput: Lens = Record.main_output
    """Selector for the main app output."""

    RecordCalls: Lens = Record.app  # type: ignore
    """Selector for the calls made by the wrapped app.

    Laid out by path into components.
    """

    RecordCall: Lens = Record.calls[-1]
    """Selector for the first called method (last to return)."""

    RecordArgs: Lens = RecordCall.args
    """Selector for the whole set of inputs/arguments to the first called / last method call."""

    RecordRets: Lens = RecordCall.rets
    """Selector for the whole output of the first called / last returned method call."""

    _PREFIXES = [
        ("Select.RecordInput", RecordInput),
        ("Select.RecordOutput", RecordOutput),
        ("Select.RecordArgs", RecordArgs),
        ("Select.RecordRets", RecordRets),
        ("Select.RecordCalls", RecordCalls),
        ("Select.RecordCall", RecordCall),
        ("Select.Record", Record),
        ("Select.App", App),
    ]
    """All prefixes/shorthands defined in this class.

    Make sure this list is sorted by longest prefix first as some prefixes are
    prefixes of others.
    """

    @staticmethod
    def path_and_method(select: Select.Lens) -> Tuple[Select.Lens, str]:
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

        if not isinstance(last, serial_utils.StepItemOrAttribute):
            raise ValueError(
                "Last part of selector is not an attribute so does not name a method."
            )

        method_name = last.get_item_or_attribute()
        path = Select.Lens(path=firsts)

        return path, method_name

    @staticmethod
    def dequalify(lens: Select.Lens) -> Select.Lens:
        """If the given selector qualifies record or app, remove that qualification."""

        if len(lens.path) == 0:
            return lens

        if Select.Record.is_prefix_of(lens) or Select.App.is_prefix_of(lens):
            return Select.Lens(path=lens.path[len(Select.Record) :])

        return lens

    @deprecation_utils.staticmethod_renamed(
        "trulens.core.app.base.App.select_context"
    )
    @staticmethod
    def context(app: Optional[Any] = None) -> Lens:
        """DEPRECATED: Select the context (retrieval step outputs) of the given
        app."""

        if app is None:
            raise ValueError("App must be given to select context.")

        return app.select_context(app)

    @staticmethod
    def for_record(lens: Select.Lens) -> Lens:
        """Add the Record prefix to the beginning of the given lens."""

        return Select.Lens(path=Select.Record.path + lens.path)

    @staticmethod
    def for_app(lens: Select.Lens) -> Lens:
        """Add the App prefix to the beginning of the given lens."""

        return Select.Lens(path=Select.App.path + lens.path)

    @staticmethod
    def render_for_dashboard(lens: Select.Lens) -> str:
        """Render the given lens for use in dashboard to help user specify feedback functions."""

        if len(lens) == 0:
            return "Select.Lens()"

        ret = ""
        rest = None

        for prefix_name, prefix_lens in Select._PREFIXES:
            if prefix_lens.is_prefix_of(lens):
                ret = prefix_name
                rest = lens.path[len(prefix_lens) :]
                break

        if rest is None:
            rest = lens.path

        for step in rest:
            ret += repr(step)

        return f"{ret}"
