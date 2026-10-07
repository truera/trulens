"""
Tests that OTEL class method instrumentation is applied once per method.

Every new recorder runs `Instrument.instrument_object` over its app's
components, and that wraps their class methods. A class method that already
carries such a wrapper must not be wrapped again, or a single call emits one
span per wrapper.
"""

import collections
import inspect
from typing import List, Tuple

import pytest
from trulens.core import instruments as core_instruments
from trulens.core import session as core_session
from trulens.core.utils import serial as serial_utils
from trulens.otel.semconv import trace as semconv_trace

import tests.util.otel_test_case

try:
    # These imports require optional dependencies to be installed.
    from langchain_core import messages as lc_messages
    from langchain_core import prompts as lc_prompts
    from langchain_core.language_models import fake as lc_fake
    from langchain_core.output_parsers import string as lc_string
    from langgraph import graph as lg_graph
    from trulens.apps.langchain import tru_chain as tru_chain_app
    from trulens.apps.langgraph import tru_graph as tru_graph_app
except Exception:
    pass

SpanType = semconv_trace.SpanAttributes.SpanType


def _wrapper_layers(func) -> int:
    """Count the wrappers stacked on top of the plain function."""
    layers = 0
    while hasattr(func, "__wrapped__"):
        layers += 1
        func = func.__wrapped__
    return layers


class _StubApp:
    """Just enough of an app for `Instrument.tracked_method_wrapper`."""

    def __init__(self) -> None:
        self.session = core_session.TruSession()


class TestOtelTrackedMethodWrapper(tests.util.otel_test_case.OtelTestCase):
    def _instrument(
        self, obj: object, cls: type, must_be_first_wrapper: bool = True
    ) -> None:
        """Instrument `obj` the way a new recorder would, with its own
        `Instrument` asking for `cls.run`."""
        app = _StubApp()
        instrument = core_instruments.Instrument(
            include_classes=[cls],
            include_methods=[
                core_instruments.InstrumentedMethod(
                    "run", cls, must_be_first_wrapper=must_be_first_wrapper
                )
            ],
            app=app,
        )
        instrument.instrument_object(obj=obj, query=serial_utils.Lens())

    def test_second_recorder_does_not_rewrap(self) -> None:
        class Component:
            def run(self) -> str:
                return "ran"

        self._instrument(Component(), Component)
        first = inspect.getattr_static(Component, "run")
        self._instrument(Component(), Component)

        self.assertIs(inspect.getattr_static(Component, "run"), first)
        self.assertEqual(_wrapper_layers(first), 1)
        self.assertEqual(Component().run(), "ran")

    def test_subclass_reuses_wrapper_of_base(self) -> None:
        class Base:
            def run(self) -> str:
                return "base"

        class Sub(Base):
            pass

        self._instrument(Base(), Base)
        base_wrapper = inspect.getattr_static(Base, "run")
        self._instrument(Sub(), Base)

        self.assertIs(inspect.getattr_static(Sub, "run"), base_wrapper)
        self.assertEqual(_wrapper_layers(base_wrapper), 1)
        self.assertEqual(Sub().run(), "base")

    def test_subclass_override_is_wrapped(self) -> None:
        class Base:
            def run(self) -> str:
                return "base"

        class Sub(Base):
            def run(self) -> str:
                return "sub"

        sub_run = Sub.__dict__["run"]
        self._instrument(Base(), Base)
        self._instrument(Sub(), Base)

        sub_wrapper = inspect.getattr_static(Sub, "run")
        self.assertIsNot(sub_wrapper, inspect.getattr_static(Base, "run"))
        self.assertEqual(_wrapper_layers(sub_wrapper), 1)
        self.assertIs(inspect.unwrap(sub_wrapper), sub_run)
        self.assertEqual(Sub().run(), "sub")

    def test_base_wrapped_after_subclass(self) -> None:
        # Instrumenting a subclass instance first wraps the method the
        # subclass inherits, and then the base. The base must still get its
        # own wrapper, so that other subclasses are recorded too, even though
        # the flag on the first wrapper is also visible on the base's plain
        # function. `must_be_first_wrapper=True` would itself skip the base
        # because of that flag, as it does without this fix.
        class Base:
            def run(self) -> str:
                return "base"

        class Sub(Base):
            pass

        class Sibling(Base):
            pass

        self._instrument(Sub(), Base, must_be_first_wrapper=False)
        base_wrapper = inspect.getattr_static(Base, "run")
        self.assertEqual(_wrapper_layers(base_wrapper), 1)

        self._instrument(Sibling(), Base, must_be_first_wrapper=False)
        self.assertIs(inspect.getattr_static(Sibling, "run"), base_wrapper)
        self.assertEqual(Sibling().run(), "base")


@pytest.mark.optional
class TestOtelTrackedMethodWrapperApps(tests.util.otel_test_case.OtelTestCase):
    @staticmethod
    def _spans(app_name: str, app_version: str) -> List[Tuple[str, str]]:
        """Name and span type of every span recorded for the app."""
        tru_session = core_session.TruSession()
        tru_session.force_flush()
        events = tru_session.connector.get_events(
            app_name=app_name, app_version=app_version
        )
        return [
            (
                record["name"],
                attributes[semconv_trace.SpanAttributes.SPAN_TYPE],
            )
            for record, attributes in zip(
                events["record"], events["record_attributes"]
            )
        ]

    @staticmethod
    def _create_chain():
        return (
            lc_prompts.PromptTemplate.from_template("Q: {q}")
            | lc_fake.FakeListLLM(responses=["a"] * 10)
            | lc_string.StrOutputParser()
        )

    @staticmethod
    def _create_graph():
        def agent(state):
            return {"messages": [lc_messages.AIMessage(content="done")]}

        workflow = lg_graph.StateGraph(lg_graph.MessagesState)
        workflow.add_node("agent", agent)
        workflow.add_edge("agent", lg_graph.END)
        workflow.set_entry_point("agent")
        return workflow.compile()

    def _record_chain(self, num_recorders: int) -> List[Tuple[str, str]]:
        """Create `num_recorders` TruChains and record a call of the last."""
        app_name = f"chain_{num_recorders}"
        recorders = []
        for i in range(num_recorders):
            chain = self._create_chain()
            recorder = tru_chain_app.TruChain(
                chain, app_name=app_name, app_version=f"v{i}"
            )
            recorders.append((recorder, chain))
        recorder, chain = recorders[-1]
        with recorder:
            chain.invoke({"q": "hi"})
        return self._spans(app_name, f"v{num_recorders - 1}")

    def _record_graph(self, num_recorders: int) -> List[Tuple[str, str]]:
        """Create `num_recorders` TruGraphs and record a call of the last."""
        app_name = f"graph_{num_recorders}"
        recorders = []
        for i in range(num_recorders):
            graph = self._create_graph()
            recorder = tru_graph_app.TruGraph(
                graph, app_name=app_name, app_version=f"v{i}"
            )
            recorders.append((recorder, graph))
        recorder, graph = recorders[-1]
        with recorder:
            graph.invoke({"messages": [lc_messages.HumanMessage(content="hi")]})
        return self._spans(app_name, f"v{num_recorders - 1}")

    def test_chain_spans_do_not_grow_with_recorders(self) -> None:
        one = collections.Counter(self._record_chain(1))
        three = collections.Counter(self._record_chain(3))

        self.assertEqual(three, one)
        names = collections.Counter(name for name, _ in three)
        for name, count in names.items():
            self.assertEqual(count, 1, f"{name} has {count} spans")
        llm = "langchain_core.language_models.fake.FakeListLLM.invoke"
        self.assertEqual(three[(llm, SpanType.GENERATION)], 1)

    def test_chain_with_repeated_component_classes(self) -> None:
        chain = (
            lc_prompts.PromptTemplate.from_template("Q: {q}")
            | lc_fake.FakeListLLM(responses=["a"] * 10)
            | lc_string.StrOutputParser()
            | lc_prompts.PromptTemplate.from_template("R: {x}")
            | lc_fake.FakeListLLM(responses=["b"] * 10)
        )
        recorder = tru_chain_app.TruChain(
            chain, app_name="repeated", app_version="v0"
        )
        with recorder:
            chain.invoke({"q": "hi"})

        names = collections.Counter(
            name for name, _ in self._spans("repeated", "v0")
        )
        prompt = "langchain_core.prompts.prompt.PromptTemplate.invoke"
        llm = "langchain_core.language_models.fake.FakeListLLM.invoke"
        self.assertEqual(names[prompt], 2)
        self.assertEqual(names[llm], 2)

    def test_graph_spans_do_not_grow_with_recorders(self) -> None:
        one = collections.Counter(self._record_graph(1))
        three = collections.Counter(self._record_graph(3))

        self.assertEqual(three, one)
        names = collections.Counter(name for name, _ in three)
        for name, count in names.items():
            self.assertEqual(count, 1, f"{name} has {count} spans")
        # The graph keeps its own node span under the record root.
        self.assertEqual(three[("graph", SpanType.GRAPH_NODE)], 1)
