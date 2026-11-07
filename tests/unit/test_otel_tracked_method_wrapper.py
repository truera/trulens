"""
Tests that OTEL class method instrumentation does not grow with recorders.

Every new recorder runs `Instrument.instrument_object` over its app's
components, and that wraps their class methods. A request that already added
a wrapper to a class method must not add another, or a single call emits one
more span for every recorder created. A more specific request, such as a
GENERATION span over a generic one, must still get its wrapper.
"""

import collections
import inspect
from typing import List, Optional, Sequence, Tuple

import pytest
from trulens.apps import app as app_module
from trulens.core import instruments as core_instruments
from trulens.core import session as core_session
from trulens.core.otel import instrument as otel_instrument
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


def _spans(app_name: str, app_version: str) -> List[Tuple[str, str]]:
    """Name and span type of every span recorded for the app."""
    tru_session = core_session.TruSession()
    tru_session.force_flush()
    events = tru_session.connector.get_events(
        app_name=app_name, app_version=app_version
    )
    return [
        (record["name"], attributes[semconv_trace.SpanAttributes.SPAN_TYPE])
        for record, attributes in zip(
            events["record"], events["record_attributes"]
        )
    ]


class _StubApp:
    """Just enough of an app for `Instrument.tracked_method_wrapper`."""

    def __init__(self) -> None:
        self.session = core_session.TruSession()


def _instrument(
    obj: object,
    cls: type,
    must_be_first_wrapper: bool = True,
    span_types: Sequence[Optional[str]] = (None,),
) -> None:
    """Instrument `obj` the way a new recorder would, with its own
    `Instrument` asking for `cls.run` once per span type, in order."""
    # `Instrument` only keeps a weak reference to its app.
    app = _StubApp()
    instrument = core_instruments.Instrument(
        include_classes=[cls],
        include_methods=[
            core_instruments.InstrumentedMethod(
                "run",
                cls,
                span_type,
                must_be_first_wrapper=must_be_first_wrapper,
            )
            for span_type in span_types
        ],
        app=app,
    )
    instrument.instrument_object(obj=obj, query=serial_utils.Lens())


class TestOtelTrackedMethodWrapper(tests.util.otel_test_case.OtelTestCase):
    def test_second_recorder_does_not_rewrap(self) -> None:
        class Component:
            def run(self) -> str:
                return "ran"

        _instrument(Component(), Component)
        first = inspect.getattr_static(Component, "run")
        _instrument(Component(), Component)

        self.assertIs(inspect.getattr_static(Component, "run"), first)
        self.assertEqual(_wrapper_layers(first), 1)
        self.assertEqual(Component().run(), "ran")

    def test_subclass_reuses_wrapper_of_base(self) -> None:
        class Base:
            def run(self) -> str:
                return "base"

        class Sub(Base):
            pass

        _instrument(Base(), Base)
        base_wrapper = inspect.getattr_static(Base, "run")
        _instrument(Sub(), Base)

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
        _instrument(Base(), Base)
        _instrument(Sub(), Base)

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

        _instrument(Sub(), Base, must_be_first_wrapper=False)
        base_wrapper = inspect.getattr_static(Base, "run")
        self.assertEqual(_wrapper_layers(base_wrapper), 1)

        _instrument(Sibling(), Base, must_be_first_wrapper=False)
        self.assertIs(inspect.getattr_static(Sibling, "run"), base_wrapper)
        self.assertEqual(Sibling().run(), "base")

    def _record_component(
        self, span_types: Sequence[Optional[str]], num_recorders: int
    ) -> List[Tuple[str, str]]:
        """Instrument a fresh component class `num_recorders` times asking
        for `span_types` in order, then record one call nested under a
        record root."""

        class Component:
            def run(self) -> str:
                return "ran"

        class Outer:
            def __init__(self) -> None:
                self.component = Component()

            @otel_instrument.instrument()
            def query(self) -> str:
                return self.component.run()

        outer = Outer()
        for _ in range(num_recorders):
            _instrument(
                outer.component,
                Component,
                must_be_first_wrapper=True,
                span_types=span_types,
            )

        app_name = f"component_{num_recorders}_{'_'.join(map(str, span_types))}"
        tru_app = app_module.TruApp(
            outer, app_name=app_name, app_version="v1", main_method=outer.query
        )
        with tru_app:
            outer.query()
        return [
            (name.rsplit(".", 1)[-1], span_type)
            for name, span_type in _spans(app_name, "v1")
            if name.endswith("Component.run")
        ]

    def test_specific_request_after_generic_wrapper(self) -> None:
        # A generic UNKNOWN wrapper made first must not hide a GENERATION
        # request that comes after it.
        span_types = (SpanType.UNKNOWN, SpanType.GENERATION)
        one = self._record_component(span_types, num_recorders=1)
        three = self._record_component(span_types, num_recorders=3)

        self.assertIn(("run", SpanType.GENERATION), one)
        self.assertEqual(collections.Counter(three), collections.Counter(one))

    def test_generic_request_after_specific_wrapper(self) -> None:
        span_types = (SpanType.GENERATION, SpanType.UNKNOWN)
        one = self._record_component(span_types, num_recorders=1)
        three = self._record_component(span_types, num_recorders=3)

        self.assertIn(("run", SpanType.GENERATION), one)
        self.assertEqual(collections.Counter(three), collections.Counter(one))


@pytest.mark.optional
class TestOtelTrackedMethodWrapperApps(tests.util.otel_test_case.OtelTestCase):
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
        return _spans(app_name, f"v{num_recorders - 1}")

    def _record_llm(self, num_recorders: int) -> List[Tuple[str, str]]:
        """Like `_record_chain` with a bare LLM as the whole app."""
        app_name = f"llm_{num_recorders}"
        recorders = []
        for i in range(num_recorders):
            llm = lc_fake.FakeListLLM(responses=["a"] * 10)
            recorder = tru_chain_app.TruChain(
                llm,
                app_name=app_name,
                app_version=f"v{i}",
                main_method=llm.invoke,
            )
            recorders.append(recorder)
        recorders[-1].instrumented_invoke_main_method(
            run_name="run", input_id="42", main_method_args=("hi",)
        )
        return _spans(app_name, f"v{num_recorders - 1}")

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
        return _spans(app_name, f"v{num_recorders - 1}")

    def test_chain_spans_do_not_grow_with_recorders(self) -> None:
        one = collections.Counter(self._record_chain(1))
        three = collections.Counter(self._record_chain(3))

        self.assertEqual(three, one)
        llm = "langchain_core.language_models.fake.FakeListLLM.invoke"
        self.assertEqual(three[(llm, SpanType.GENERATION)], 1)

    def test_llm_app_spans_do_not_grow_with_recorders(self) -> None:
        one = collections.Counter(self._record_llm(1))
        three = collections.Counter(self._record_llm(3))

        self.assertEqual(three, one)
        llm = "langchain_core.language_models.fake.FakeListLLM.invoke"
        self.assertEqual(three[(llm, SpanType.GENERATION)], 1)

    def test_chain_with_repeated_component_classes(self) -> None:
        single = collections.Counter(self._record_chain(1))

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
        repeated = collections.Counter(_spans("repeated", "v0"))

        # Each of the two prompts and two LLMs is recorded like the single
        # one in a chain with one of each.
        for name in (
            "langchain_core.prompts.prompt.PromptTemplate.invoke",
            "langchain_core.language_models.fake.FakeListLLM.invoke",
        ):
            for key, count in single.items():
                if key[0] == name:
                    self.assertEqual(repeated[key], 2 * count, key)

    def test_graph_spans_do_not_grow_with_recorders(self) -> None:
        one = collections.Counter(self._record_graph(1))
        three = collections.Counter(self._record_graph(3))

        self.assertEqual(three, one)
        names = collections.Counter(name for name, _ in three)
        for name, count in names.items():
            self.assertEqual(count, 1, f"{name} has {count} spans")
        # The graph keeps its own node span under the record root.
        self.assertEqual(three[("graph", SpanType.GRAPH_NODE)], 1)
