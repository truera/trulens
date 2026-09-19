"""
Tests for OTEL TruAutoGen app.

The agents here reply from registered reply functions rather than from an LLM,
so the whole conversation flow -- agent turns, tool calls, group chat speaker
selection -- is exercised without any network access.
"""

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytest
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import GenAIAttributes
from trulens.otel.semconv.trace import SpanAttributes

import tests.util.otel_tru_app_test_case

try:
    from autogen import ConversableAgent
    from autogen import GroupChat
    from autogen import GroupChatManager
    from trulens.apps.autogen import AutoGenInstrument
    from trulens.apps.autogen import TruAutoGen
except Exception:
    pass


def _canned_agent(name: str, reply: Any, **kwargs: Any) -> "ConversableAgent":
    """An agent that always replies with `reply`, without calling an LLM."""

    agent = ConversableAgent(
        name,
        llm_config=False,
        human_input_mode="NEVER",
        system_message=f"You are {name}.",
        **kwargs,
    )
    agent.register_reply(
        [ConversableAgent, None],
        lambda self, messages=None, sender=None, config=None: (True, reply),
    )
    return agent


def _user_proxy(name: str = "user") -> "ConversableAgent":
    return ConversableAgent(
        name,
        llm_config=False,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
    )


@pytest.mark.optional
class TestOtelTruAutoGen(tests.util.otel_tru_app_test_case.OtelTruAppTestCase):
    @staticmethod
    def _create_test_app_info() -> (
        tests.util.otel_tru_app_test_case.TestAppInfo
    ):
        app = _user_proxy()
        return tests.util.otel_tru_app_test_case.TestAppInfo(
            app=app, main_method=app.initiate_chat, TruAppClass=TruAutoGen
        )

    def _spans(self) -> pd.DataFrame:
        TruSession().force_flush()
        return self._get_events()

    @staticmethod
    def _attributes_of_type(
        events: pd.DataFrame, span_type: SpanAttributes.SpanType
    ) -> List[Dict[str, Any]]:
        return [
            attributes
            for attributes in events["record_attributes"]
            if attributes.get(SpanAttributes.SPAN_TYPE) == span_type.value
        ]

    @staticmethod
    def _span_tree(
        events: pd.DataFrame,
    ) -> Dict[str, Tuple[Optional[str], str]]:
        """Map span id -> (parent span id, span type)."""

        return {
            row["trace"]["span_id"]: (
                row["trace"].get("parent_id"),
                row["record_attributes"].get(SpanAttributes.SPAN_TYPE),
            )
            for _, row in events.iterrows()
        }

    @staticmethod
    def _span_names(events: pd.DataFrame) -> List[str]:
        return [row["record"]["name"] for _, row in events.iterrows()]

    def _parent_type_of(
        self, events: pd.DataFrame, span_type: SpanAttributes.SpanType
    ) -> List[Optional[str]]:
        tree = self._span_tree(events)
        parent_types = []
        for _, (parent_id, this_type) in tree.items():
            if this_type != span_type.value:
                continue
            parent = tree.get(parent_id)
            parent_types.append(parent[1] if parent else None)
        return parent_types

    def test_two_agent_conversation(self) -> None:
        """A two-agent chat records a record root and the replier's turn."""

        assistant = _canned_agent(
            "assistant", "The sky is blue because of Rayleigh scattering."
        )
        user = _user_proxy()

        tru_recorder = TruAutoGen(
            user,
            app_name="two_agent",
            app_version="v1",
            main_method=user.initiate_chat,
        )

        with tru_recorder:
            result = user.initiate_chat(
                assistant, message="Why is the sky blue?", max_turns=1
            )

        self.assertIn("Rayleigh", result.summary)

        events = self._spans()

        record_roots = self._attributes_of_type(
            events, SpanAttributes.SpanType.RECORD_ROOT
        )
        self.assertEqual(1, len(record_roots))
        self.assertEqual(
            "Why is the sky blue?",
            record_roots[0][SpanAttributes.RECORD_ROOT.INPUT],
        )
        self.assertEqual(
            "The sky is blue because of Rayleigh scattering.",
            record_roots[0][SpanAttributes.RECORD_ROOT.OUTPUT],
        )

        agent_spans = self._attributes_of_type(
            events, SpanAttributes.SpanType.AGENT
        )
        self.assertEqual(1, len(agent_spans))
        agent_span = agent_spans[0]
        self.assertEqual("assistant", agent_span[SpanAttributes.AGENT.NAME])
        self.assertEqual(
            "assistant", agent_span[SpanAttributes.WORKFLOW.AGENT_NAME]
        )
        self.assertEqual(
            "You are assistant.",
            agent_span[SpanAttributes.AGENT.SYSTEM_MESSAGE],
        )
        self.assertEqual(
            "Why is the sky blue?",
            agent_span[SpanAttributes.AGENT.INPUT_MESSAGE],
        )
        self.assertEqual(
            "The sky is blue because of Rayleigh scattering.",
            agent_span[SpanAttributes.AGENT.OUTPUT_MESSAGE],
        )
        # The turn's whole history is captured, not only the latest message.
        self.assertIn(
            "Why is the sky blue?",
            agent_span[SpanAttributes.AGENT.INPUT_MESSAGES],
        )

    def test_recording_without_a_main_method(self) -> None:
        """Without `main_method`, the outermost AutoGen call starts the record."""

        assistant = _canned_agent("assistant", "Because of scattering.")
        user = _user_proxy()

        tru_recorder = TruAutoGen(
            user, app_name="no_main_method", app_version="v1"
        )

        with tru_recorder:
            user.initiate_chat(
                assistant, message="Why is the sky blue?", max_turns=1
            )

        record_roots = self._attributes_of_type(
            self._spans(), SpanAttributes.SpanType.RECORD_ROOT
        )
        self.assertEqual(1, len(record_roots))
        self.assertEqual(
            "Why is the sky blue?",
            record_roots[0][SpanAttributes.RECORD_ROOT.INPUT],
        )
        self.assertEqual(
            "Because of scattering.",
            record_roots[0][SpanAttributes.RECORD_ROOT.OUTPUT],
        )

    def test_async_conversation(self) -> None:
        """The async entry points record the same spans as the sync ones."""

        assistant = _canned_agent("assistant", "An async reply.")
        user = _user_proxy()

        tru_recorder = TruAutoGen(
            user,
            app_name="async_chat",
            app_version="v1",
            main_method=user.a_initiate_chat,
        )

        async def run():
            with tru_recorder:
                return await user.a_initiate_chat(
                    assistant, message="Anything there?", max_turns=1
                )

        result = asyncio.run(run())
        self.assertIn("An async reply.", result.summary)

        events = self._spans()

        record_roots = self._attributes_of_type(
            events, SpanAttributes.SpanType.RECORD_ROOT
        )
        self.assertEqual(1, len(record_roots))
        self.assertEqual(
            "Anything there?",
            record_roots[0][SpanAttributes.RECORD_ROOT.INPUT],
        )
        self.assertEqual(
            "An async reply.",
            record_roots[0][SpanAttributes.RECORD_ROOT.OUTPUT],
        )

        agent_spans = self._attributes_of_type(
            events, SpanAttributes.SpanType.AGENT
        )
        self.assertEqual(1, len(agent_spans))
        self.assertEqual("assistant", agent_spans[0][SpanAttributes.AGENT.NAME])
        self.assertEqual(
            "An async reply.",
            agent_spans[0][SpanAttributes.AGENT.OUTPUT_MESSAGE],
        )
        self.assertIn("assistant.a_generate_reply", self._span_names(events))

    def test_agents_other_than_the_wrapped_one_are_recorded(self) -> None:
        """Wrapping one agent records every agent in the conversation.

        The recipient is not reachable from the wrapped agent, so this is what
        class-level instrumentation buys.
        """

        first = _canned_agent("first", "First reply.")
        second = _canned_agent("second", "Second reply.")
        user = _user_proxy()

        tru_recorder = TruAutoGen(
            user,
            app_name="unreachable_agents",
            app_version="v1",
            main_method=user.initiate_chat,
        )

        with tru_recorder:
            user.initiate_chat(first, message="Hello", max_turns=1)
            user.initiate_chat(second, message="Hello again", max_turns=1)

        agent_names = {
            attributes[SpanAttributes.AGENT.NAME]
            for attributes in self._attributes_of_type(
                self._spans(), SpanAttributes.SpanType.AGENT
            )
        }
        self.assertEqual({"first", "second"}, agent_names)

    def test_tool_call_nests_under_the_agent_that_ran_it(self) -> None:
        """A tool execution is a TOOL span beneath the agent's AGENT span."""

        def word_count(text: str) -> int:
            return len(text.split())

        executor = ConversableAgent(
            "executor",
            llm_config=False,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )
        executor.register_function({"word_count": word_count})

        tool_call_message = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "word_count",
                        "arguments": '{"text": "one two three"}',
                    },
                }
            ],
        }
        caller = _user_proxy("caller")

        tru_recorder = TruAutoGen(
            caller,
            app_name="tool_call",
            app_version="v1",
            main_method=caller.initiate_chat,
        )

        with tru_recorder:
            caller.initiate_chat(
                executor, message=tool_call_message, max_turns=1
            )

        events = self._spans()

        tool_spans = self._attributes_of_type(
            events, SpanAttributes.SpanType.TOOL
        )
        self.assertEqual(1, len(tool_spans))
        tool_span = tool_spans[0]
        self.assertEqual("word_count", tool_span[GenAIAttributes.TOOL.NAME])
        self.assertEqual("call_1", tool_span[GenAIAttributes.TOOL.CALL_ID])
        self.assertEqual(
            '{"text": "one two three"}',
            tool_span[GenAIAttributes.TOOL.CALL_ARGUMENTS],
        )
        self.assertEqual("3", tool_span[GenAIAttributes.TOOL.CALL_RESULT])
        self.assertNotIn(SpanAttributes.CALL.ERROR, tool_span)
        self.assertIn("execute_function: word_count", self._span_names(events))

        self.assertEqual(
            [SpanAttributes.SpanType.AGENT.value],
            self._parent_type_of(events, SpanAttributes.SpanType.TOOL),
        )

    def test_failed_tool_call_is_recorded_as_an_error(self) -> None:
        """AutoGen reports an unknown tool as a result, not an exception."""

        executor = ConversableAgent(
            "executor",
            llm_config=False,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )
        caller = _user_proxy("caller")

        tru_recorder = TruAutoGen(
            caller,
            app_name="failed_tool",
            app_version="v1",
            main_method=caller.initiate_chat,
        )

        with tru_recorder:
            caller.initiate_chat(
                executor,
                message={
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "missing", "arguments": "{}"},
                        }
                    ],
                },
                max_turns=1,
            )

        tool_spans = self._attributes_of_type(
            self._spans(), SpanAttributes.SpanType.TOOL
        )
        self.assertEqual(1, len(tool_spans))
        self.assertEqual("missing", tool_spans[0][GenAIAttributes.TOOL.NAME])
        self.assertIn(
            "not found", tool_spans[0][GenAIAttributes.TOOL.CALL_RESULT]
        )
        self.assertIn("not found", tool_spans[0][SpanAttributes.CALL.ERROR])

    def test_group_chat_span_tree(self) -> None:
        """A group chat records a speaker selection and a reply per round."""

        researcher = _canned_agent("researcher", "Findings.")
        writer = _canned_agent("writer", "Article.")
        user = _user_proxy()

        group_chat = GroupChat(
            agents=[researcher, writer],
            messages=[],
            max_round=3,
            speaker_selection_method="round_robin",
        )
        manager = GroupChatManager(groupchat=group_chat, llm_config=False)

        tru_recorder = TruAutoGen(
            user,
            app_name="group_chat",
            app_version="v1",
            main_method=user.initiate_chat,
        )

        with tru_recorder:
            user.initiate_chat(manager, message="Write about the sky.")

        events = self._spans()

        selection_spans = self._attributes_of_type(
            events, SpanAttributes.SpanType.WORKFLOW_STEP
        )
        # One selection per round, plus the initiate_chat span itself, which
        # is the record root and so is not a workflow step.
        self.assertEqual(2, len(selection_spans))
        for selection_span in selection_spans:
            self.assertEqual(
                "round_robin",
                selection_span[
                    SpanAttributes.WORKFLOW.SPEAKER_SELECTION_METHOD
                ],
            )
            self.assertEqual(
                ("researcher", "writer"),
                tuple(
                    selection_span[
                        SpanAttributes.WORKFLOW.PARTICIPANT_AGENT_NAMES
                    ]
                ),
            )

        self.assertEqual(
            ["researcher", "writer"],
            [
                selection_span[SpanAttributes.WORKFLOW.AGENT_NAME]
                for selection_span in selection_spans
            ],
        )

        agent_names = [
            attributes[SpanAttributes.AGENT.NAME]
            for attributes in self._attributes_of_type(
                events, SpanAttributes.SpanType.AGENT
            )
        ]
        # The manager takes a turn, and each selected speaker replies within it.
        self.assertEqual(
            ["chat_manager", "researcher", "writer"], sorted(agent_names)
        )

        # Spans are named after what they did, so the tree reads as the
        # conversation did.
        self.assertEqual(
            [
                "autogen.agentchat.conversable_agent.ConversableAgent.initiate_chat",
                "chat_manager.generate_reply",
                "select_speaker: researcher",
                "researcher.generate_reply",
                "select_speaker: writer",
                "writer.generate_reply",
            ],
            self._span_names(events),
        )

    def test_main_input_and_output(self) -> None:
        """`main_input` / `main_output` read AutoGen's own message shapes."""

        agent = _user_proxy()
        tru_recorder = TruAutoGen(agent, app_name="main_io", app_version="v1")

        def initiate_chat(recipient, message=None):
            pass

        sig = inspect.signature(initiate_chat)

        self.assertEqual(
            "plain string",
            tru_recorder.main_input(
                initiate_chat,
                sig,
                sig.bind_partial(recipient=None, message="plain string"),
            ),
        )
        self.assertEqual(
            "from a dict",
            tru_recorder.main_input(
                initiate_chat,
                sig,
                sig.bind_partial(
                    recipient=None,
                    message={"role": "user", "content": "from a dict"},
                ),
            ),
        )

        def generate_reply(messages=None):
            pass

        reply_sig = inspect.signature(generate_reply)
        self.assertEqual(
            "latest",
            tru_recorder.main_input(
                generate_reply,
                reply_sig,
                reply_sig.bind_partial(
                    messages=[
                        {"role": "user", "content": "earlier"},
                        {"role": "user", "content": "latest"},
                    ]
                ),
            ),
        )

        self.assertEqual(
            "a reply",
            tru_recorder.main_output(
                generate_reply,
                reply_sig,
                reply_sig.bind_partial(messages=[]),
                {"role": "assistant", "content": "a reply"},
            ),
        )

    def test_llm_call_is_a_generation_span(self) -> None:
        """An agent's LLM call is a GENERATION span inside its turn.

        Uses AutoGen's custom model client hook so nothing leaves the process.
        """

        class FakeClient:
            def __init__(self, config, **kwargs):
                pass

            def create(self, params):
                return SimpleNamespace(
                    model="fake-model",
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="A canned answer.", tool_calls=None
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=10, completion_tokens=5, total_tokens=15
                    ),
                    cost=0.0,
                )

            def message_retrieval(self, response):
                return [choice.message.content for choice in response.choices]

            def cost(self, response):
                return 0.0

            @staticmethod
            def get_usage(response):
                return {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "cost": 0.0,
                    "model": "fake-model",
                }

        assistant = ConversableAgent(
            "assistant",
            llm_config={
                "config_list": [
                    {"model": "fake-model", "model_client_cls": "FakeClient"}
                ]
            },
            human_input_mode="NEVER",
        )
        assistant.register_model_client(model_client_cls=FakeClient)
        user = _user_proxy()

        tru_recorder = TruAutoGen(
            user,
            app_name="generation",
            app_version="v1",
            main_method=user.initiate_chat,
        )

        with tru_recorder:
            user.initiate_chat(assistant, message="Hello", max_turns=1)

        events = self._spans()

        generation_spans = self._attributes_of_type(
            events, SpanAttributes.SpanType.GENERATION
        )
        self.assertEqual(1, len(generation_spans))
        self.assertEqual(
            "fake-model", generation_spans[0][SpanAttributes.COST.MODEL]
        )
        self.assertEqual(
            [SpanAttributes.SpanType.AGENT.value],
            self._parent_type_of(events, SpanAttributes.SpanType.GENERATION),
        )

    def test_instrumentation_spec(self) -> None:
        """The methods the issue calls for are all in the spec."""

        spec = {
            (method.method, method.class_filter.__name__): method.span_type
            for method in AutoGenInstrument.Default.METHODS()
        }

        self.assertEqual(
            SpanAttributes.SpanType.AGENT,
            spec[("generate_reply", "ConversableAgent")],
        )
        self.assertEqual(
            SpanAttributes.SpanType.AGENT,
            spec[("a_generate_reply", "ConversableAgent")],
        )
        self.assertEqual(
            SpanAttributes.SpanType.TOOL,
            spec[("execute_function", "ConversableAgent")],
        )
        self.assertEqual(
            SpanAttributes.SpanType.WORKFLOW_STEP,
            spec[("select_speaker", "GroupChat")],
        )
        self.assertEqual(
            SpanAttributes.SpanType.GENERATION,
            spec[("create", "OpenAIWrapper")],
        )

        self.assertIn(ConversableAgent, AutoGenInstrument.Default.CLASSES())
        self.assertIn(GroupChat, AutoGenInstrument.Default.CLASSES())

    def test_session_auto_detection(self) -> None:
        """`TruSession.App` recognizes an AutoGen agent."""

        agent = _user_proxy()
        tru_app = TruSession().App(agent, app_name="detected")

        self.assertIsInstance(tru_app, TruAutoGen)
        self.assertEqual("detected", tru_app.app_name)
