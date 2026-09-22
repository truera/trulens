"""AutoGen (AG2) instrumentation.

Records an AutoGen conversation as a span tree: every agent turn becomes an
`AGENT` span, every tool execution a `TOOL` span, every group chat speaker
selection a `WORKFLOW_STEP` span, and every LLM call made on an agent's behalf
a `GENERATION` span.
"""

from __future__ import annotations

from inspect import BoundArguments
from inspect import Signature
import json
import logging
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from opentelemetry.trace import get_current_span
from pydantic import Field
from trulens.core import app as core_app
from trulens.core import instruments as core_instruments
from trulens.core.instruments import InstrumentedMethod
from trulens.core.otel.instrument import instrument_method
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.core.session import TruSession
from trulens.core.utils import pyschema as pyschema_utils
from trulens.experimental.otel_tracing.core.span import Attributes
from trulens.otel.semconv.trace import GenAIAttributes
from trulens.otel.semconv.trace import SpanAttributes

from autogen import ConversableAgent
from autogen import GroupChat
from autogen import GroupChatManager
from autogen.agentchat.chat import ChatResult

try:
    from autogen.oai.client import OpenAIWrapper
except ImportError:  # pragma: no cover - depends on the autogen build
    OpenAIWrapper = None

logger = logging.getLogger(__name__)

MAX_INSTRUMENTED_MESSAGES = 50
"""Most recent messages kept when serializing a conversation history.

A long-running group chat replays its whole history on every turn, so keeping
all of it would grow each span quadratically in the number of turns.
"""


def _to_text(value: Any) -> Optional[str]:
    """Render `value` as a span-attribute-friendly string."""

    if value is None:
        return None

    if isinstance(value, str):
        return value

    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)


def _update_span_name(span_name: str) -> None:
    """Name the current span after what it is doing, when possible."""

    try:
        current_span = get_current_span()
        if current_span is not None and hasattr(current_span, "update_name"):
            current_span.update_name(span_name)
    except Exception:
        logger.warning("Could not rename span to %s.", span_name, exc_info=True)


def _message_role(message: Any) -> Optional[str]:
    """Role of an AutoGen message, if it carries one."""

    if isinstance(message, dict):
        role = message.get("role")
        return role if isinstance(role, str) else None

    role = getattr(message, "role", None)
    return role if isinstance(role, str) else None


def _message_content(message: Any) -> Optional[str]:
    """Human-readable content of an AutoGen message.

    AutoGen messages are dicts with a `content` entry, but a tool call has
    `content=None` and carries the request in `tool_calls` or `function_call`
    instead, and multi-modal content is a list of parts.
    """

    if message is None:
        return None

    if isinstance(message, str):
        return message

    if not isinstance(message, dict):
        content = getattr(message, "content", None)
        return _to_text(content) if content is not None else None

    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    if content is not None and not isinstance(content, str):
        return _to_text(content)

    for call_field in ("tool_calls", "function_call"):
        call = message.get(call_field)
        if call:
            return _to_text(call)

    return content if isinstance(content, str) else None


def _messages(value: Any) -> List[Any]:
    """Coerce whatever AutoGen passed as `messages` into a list."""

    if value is None:
        return []
    if isinstance(value, dict) or isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _serialize_messages(messages: Any) -> Optional[str]:
    """Serialize a message history, keeping only the most recent messages."""

    history = _messages(messages)
    if not history:
        return None

    return _to_text([
        {
            "role": _message_role(message),
            "content": _message_content(message),
        }
        for message in history[-MAX_INSTRUMENTED_MESSAGES:]
    ])


def _agent_name(agent: Any) -> Optional[str]:
    name = getattr(agent, "name", None)
    return name if isinstance(name, str) else None


def _system_message(agent: Any) -> Optional[str]:
    """System message of an agent.

    `ConversableAgent.system_message` is a property that reads the first entry
    of `_oai_system_message`, so it is absent on agents that have no LLM
    config.
    """

    try:
        system_message = getattr(agent, "system_message", None)
    except Exception:
        return None

    return _to_text(system_message) if system_message else None


def _agent_span(method_name: str) -> Tuple[SpanAttributes.SpanType, Attributes]:
    """An agent producing a reply."""

    def _attributes(ret, exception, *args, **kwargs) -> Dict[str, Any]:
        agent = args[0] if args else None
        attributes: Dict[str, Any] = {}

        name = _agent_name(agent)
        if name is not None:
            attributes[SpanAttributes.AGENT.NAME] = name
            # Also emitted as a workflow attribute so that agent turns from
            # any framework can be selected the same way.
            attributes[SpanAttributes.WORKFLOW.AGENT_NAME] = name
            _update_span_name(f"{name}.{method_name}")

        system_message = _system_message(agent)
        if system_message is not None:
            attributes[SpanAttributes.AGENT.SYSTEM_MESSAGE] = system_message

        # AutoGen defaults an agent's description to its system message, so
        # only record it when it actually says something else.
        description = getattr(agent, "description", None)
        if (
            isinstance(description, str)
            and description
            and description != system_message
        ):
            attributes[SpanAttributes.AGENT.DESCRIPTION] = description

        messages = kwargs.get("messages")
        if messages is None and agent is not None:
            # `generate_reply` falls back to the history the agent holds for
            # the sender when it is called without explicit messages.
            messages = _chat_history(agent, kwargs.get("sender"))

        history = _messages(messages)
        serialized = _serialize_messages(history)
        if serialized is not None:
            attributes[SpanAttributes.AGENT.INPUT_MESSAGES] = serialized
            attributes[SpanAttributes.WORKFLOW.INPUT_EVENT] = serialized

        if history:
            latest_content = _message_content(history[-1])
            if latest_content is not None:
                attributes[SpanAttributes.AGENT.INPUT_MESSAGE] = latest_content
            latest_role = _message_role(history[-1])
            if latest_role is not None:
                attributes[SpanAttributes.AGENT.INPUT_MESSAGE_ROLE] = (
                    latest_role
                )

        reply = _message_content(ret)
        if reply is not None:
            attributes[SpanAttributes.AGENT.OUTPUT_MESSAGE] = reply
            attributes[SpanAttributes.WORKFLOW.OUTPUT_EVENT] = reply

        if exception is not None:
            attributes[SpanAttributes.AGENT.ERROR] = str(exception)
            attributes[SpanAttributes.WORKFLOW.ERROR] = str(exception)

        return attributes

    return SpanAttributes.SpanType.AGENT, _attributes


def _chat_history(agent: Any, sender: Any) -> Any:
    """Messages `agent` holds for `sender`, or its whole history."""

    chat_messages = getattr(agent, "chat_messages", None)
    if not chat_messages:
        return None

    try:
        if sender is not None and sender in chat_messages:
            return chat_messages[sender]
        if len(chat_messages) == 1:
            return next(iter(chat_messages.values()))
    except Exception:
        logger.debug("Could not read chat history off agent.", exc_info=True)

    return None


def _tool_span(method_name: str) -> Tuple[SpanAttributes.SpanType, Attributes]:
    """An agent executing a registered function or tool."""

    def _attributes(ret, exception, *args, **kwargs) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        agent_name = _agent_name(args[0] if args else None)
        if agent_name is not None:
            attributes[SpanAttributes.AGENT.NAME] = agent_name
            attributes[SpanAttributes.WORKFLOW.AGENT_NAME] = agent_name

        func_call = kwargs.get("func_call")
        if isinstance(func_call, dict):
            tool_name = func_call.get("name")
            if isinstance(tool_name, str):
                attributes[GenAIAttributes.TOOL.NAME] = tool_name
                _update_span_name(f"{method_name}: {tool_name}")
            arguments = func_call.get("arguments")
            if arguments is not None:
                attributes[GenAIAttributes.TOOL.CALL_ARGUMENTS] = _to_text(
                    arguments
                )

        call_id = kwargs.get("call_id")
        if isinstance(call_id, str):
            attributes[GenAIAttributes.TOOL.CALL_ID] = call_id

        # `execute_function` returns `(is_exec_success, message)`. A failed
        # call does not raise: the failure is reported back to the agent as
        # the tool result.
        if isinstance(ret, tuple) and len(ret) == 2:
            is_success, message = ret
            result = _message_content(message)
            if result is not None:
                attributes[GenAIAttributes.TOOL.CALL_RESULT] = result
                if not is_success:
                    attributes[SpanAttributes.CALL.ERROR] = result

        return attributes

    return SpanAttributes.SpanType.TOOL, _attributes


def _speaker_selection_span(
    method_name: str,
) -> Tuple[SpanAttributes.SpanType, Attributes]:
    """A group chat choosing which agent speaks next."""

    def _attributes(ret, exception, *args, **kwargs) -> Dict[str, Any]:
        group_chat = args[0] if args else None
        attributes: Dict[str, Any] = {}

        method = getattr(group_chat, "speaker_selection_method", None)
        if isinstance(method, str):
            attributes[SpanAttributes.WORKFLOW.SPEAKER_SELECTION_METHOD] = (
                method
            )
        elif method is not None:
            attributes[SpanAttributes.WORKFLOW.SPEAKER_SELECTION_METHOD] = (
                getattr(method, "__name__", str(method))
            )

        participants = [
            name
            for name in (
                _agent_name(agent)
                for agent in getattr(group_chat, "agents", None) or []
            )
            if name is not None
        ]
        if participants:
            attributes[SpanAttributes.WORKFLOW.PARTICIPANT_AGENT_NAMES] = (
                participants
            )

        last_speaker = _agent_name(kwargs.get("last_speaker"))
        if last_speaker is not None:
            attributes[SpanAttributes.WORKFLOW.INPUT_EVENT] = last_speaker

        selected = _agent_name(ret)
        if selected is not None:
            _update_span_name(f"{method_name}: {selected}")
            # The selected speaker is what this step produced, and is also the
            # agent the nested reply spans belong to.
            attributes[SpanAttributes.WORKFLOW.AGENT_NAME] = selected
            attributes[SpanAttributes.WORKFLOW.OUTPUT_EVENT] = selected

        if exception is not None:
            attributes[SpanAttributes.WORKFLOW.ERROR] = str(exception)

        return attributes

    return SpanAttributes.SpanType.WORKFLOW_STEP, _attributes


def _generation_span() -> Tuple[SpanAttributes.SpanType, Attributes]:
    """An LLM call made on an agent's behalf.

    Token counts and costs are deliberately left to the provider cost
    computers, which write them onto whichever span is current -- this one.
    Setting them here as well would double them, because TruLens sums cost
    attributes that arrive on a span more than once.
    """

    def _attributes(ret, exception, *args, **kwargs) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        model = getattr(ret, "model", None) or kwargs.get("model")
        if isinstance(model, str):
            attributes[SpanAttributes.COST.MODEL] = model

        return attributes

    return SpanAttributes.SpanType.GENERATION, _attributes


def _chat_span() -> Tuple[SpanAttributes.SpanType, Attributes]:
    """One agent starting a conversation with another.

    The outermost such call in a recording becomes the record root; nested
    chats become workflow steps beneath it.
    """

    def _attributes(ret, exception, *args, **kwargs) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        initiator = _agent_name(args[0] if args else None)
        if initiator is not None:
            attributes[SpanAttributes.WORKFLOW.AGENT_NAME] = initiator

        participants = [
            name
            for name in (initiator, _agent_name(kwargs.get("recipient")))
            if name is not None
        ]
        if participants:
            attributes[SpanAttributes.WORKFLOW.PARTICIPANT_AGENT_NAMES] = (
                participants
            )

        message = _message_content(kwargs.get("message"))
        if message is not None:
            attributes[SpanAttributes.WORKFLOW.INPUT_EVENT] = message

        if isinstance(ret, ChatResult):
            # The per-turn detail lives on the nested agent spans; the chat
            # span carries the answer the conversation arrived at.
            summary = _chat_result_summary(ret)
            if summary is not None:
                attributes[SpanAttributes.WORKFLOW.OUTPUT_EVENT] = summary

        if exception is not None:
            attributes[SpanAttributes.WORKFLOW.ERROR] = str(exception)

        return attributes

    return SpanAttributes.SpanType.WORKFLOW_STEP, _attributes


def _chat_result_summary(chat_result: ChatResult) -> Optional[str]:
    """Best single-string answer a finished chat produced."""

    summary = getattr(chat_result, "summary", None)
    if summary:
        return _to_text(summary)

    history = getattr(chat_result, "chat_history", None) or []
    for message in reversed(history):
        content = _message_content(message)
        if content:
            return content

    return None


class AutoGenInstrument(core_instruments.Instrument):
    """Instrumentation for AutoGen apps."""

    class Default:
        """Instrumentation specification for AutoGen apps."""

        MODULES = {"autogen"}
        """Modules by prefix to instrument."""

        CLASSES = lambda: {  # noqa: E731
            ConversableAgent,
            GroupChat,
            GroupChatManager,
        }.union({OpenAIWrapper} if OpenAIWrapper is not None else set())
        """Classes to instrument."""

        @staticmethod
        def METHODS() -> List[InstrumentedMethod]:
            methods = [
                InstrumentedMethod(
                    "initiate_chat", ConversableAgent, *_chat_span()
                ),
                InstrumentedMethod(
                    "a_initiate_chat", ConversableAgent, *_chat_span()
                ),
                InstrumentedMethod(
                    "generate_reply",
                    ConversableAgent,
                    *_agent_span("generate_reply"),
                ),
                InstrumentedMethod(
                    "a_generate_reply",
                    ConversableAgent,
                    *_agent_span("a_generate_reply"),
                ),
                InstrumentedMethod(
                    "execute_function",
                    ConversableAgent,
                    *_tool_span("execute_function"),
                ),
                InstrumentedMethod(
                    "a_execute_function",
                    ConversableAgent,
                    *_tool_span("a_execute_function"),
                ),
                InstrumentedMethod(
                    "select_speaker",
                    GroupChat,
                    *_speaker_selection_span("select_speaker"),
                ),
                InstrumentedMethod(
                    "a_select_speaker",
                    GroupChat,
                    *_speaker_selection_span("a_select_speaker"),
                ),
            ]

            if OpenAIWrapper is not None:
                methods.append(
                    InstrumentedMethod(
                        "create", OpenAIWrapper, *_generation_span()
                    )
                )

            return methods

        """Methods to instrument."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            include_modules=AutoGenInstrument.Default.MODULES,
            include_classes=AutoGenInstrument.Default.CLASSES(),
            include_methods=AutoGenInstrument.Default.METHODS(),
            *args,
            **kwargs,
        )


class TruAutoGen(core_app.App):
    """Recorder for _AutoGen_ (_AG2_) applications.

    Wrapping any one agent instruments the AutoGen classes, so every agent
    taking part in the conversation is recorded -- including agents created
    after the recorder, and the manager and members of a group chat.

    Example: "Recording a two-agent conversation"

        ```python
        from autogen import AssistantAgent, UserProxyAgent
        from trulens.apps.autogen import TruAutoGen

        assistant = AssistantAgent("assistant", llm_config=llm_config)
        user = UserProxyAgent("user", human_input_mode="NEVER")

        tru_recorder = TruAutoGen(
            user,
            app_name="debate",
            app_version="v1",
            main_method=user.initiate_chat,
        )

        with tru_recorder as recording:
            result = user.initiate_chat(assistant, message="Why is the sky blue?")
        ```

    Example: "Evaluating individual agent replies"

        Each agent turn is an `AGENT` span carrying the agent's name, the
        messages it was given, and the reply it produced, so a metric can be
        pointed at agent replies rather than at the conversation as a whole.

        ```python
        from trulens.core import Feedback
        from trulens.core.feedback.selector import Selector
        from trulens.otel.semconv.trace import SpanAttributes

        f_coherence = (
            Feedback(provider.coherence_with_cot_reasons, name="Coherence")
            .on({
                "text": Selector(
                    span_type=SpanAttributes.SpanType.AGENT,
                    span_attribute=SpanAttributes.AGENT.OUTPUT_MESSAGE,
                ),
            })
        )
        ```

    Args:
        app: An AutoGen agent. Usually the agent that starts the conversation,
            or the `GroupChatManager` of a group chat.

        main_method: The method that starts a conversation, such as
            `agent.initiate_chat`. Optional: without it, the outermost
            instrumented AutoGen call in the recording starts the record.

        **kwargs: Additional arguments to pass to [App][trulens.core.app.App]
            and [AppDefinition][trulens.core.schema.app.AppDefinition].
    """

    app: ConversableAgent

    root_callable: ClassVar[pyschema_utils.FunctionOrMethod] = Field(None)

    _class_instrumentation_applied: ClassVar[bool] = False

    def __init__(
        self,
        app: ConversableAgent,
        main_method: Optional[Callable] = None,
        **kwargs: Any,
    ):
        self._ensure_class_instrumentation()

        kwargs["app"] = app

        # Create `TruSession` if not already created.
        if "connector" in kwargs:
            TruSession(connector=kwargs["connector"])
        else:
            TruSession()

        if main_method is not None:
            kwargs["main_method"] = main_method

        kwargs["root_class"] = pyschema_utils.Class.of_object(app)
        kwargs["instrument"] = AutoGenInstrument(app=self)

        super().__init__(**kwargs)

    @classmethod
    def _ensure_class_instrumentation(cls) -> None:
        """Instrument the AutoGen classes themselves, once.

        The agents an app talks to are not reachable from the wrapped agent --
        a recipient is passed to `initiate_chat`, and a group chat creates
        replies from agents the recorder never sees -- so instrumenting the
        classes is what makes the whole conversation show up rather than only
        the wrapped agent's own turns.
        """

        if cls._class_instrumentation_applied:
            return

        if not is_otel_tracing_enabled():
            logger.debug(
                "OTEL tracing is disabled; skipping AutoGen class-level"
                " instrumentation."
            )
            return

        for instrumented_method in AutoGenInstrument.Default.METHODS():
            target = instrumented_method.class_filter
            if not isinstance(target, type):
                continue
            if not hasattr(target, instrumented_method.method):
                continue

            try:
                instrument_method(
                    cls=target,
                    method_name=instrumented_method.method,
                    span_type=instrumented_method.span_type,
                    attributes=instrumented_method.attributes,
                    must_be_first_wrapper=True,
                )
            except Exception:
                logger.warning(
                    "Failed to instrument %s.%s",
                    target.__name__,
                    instrumented_method.method,
                    exc_info=True,
                )

        cls._class_instrumentation_applied = True

    # App override:
    def main_input(
        self, func: Callable, sig: Signature, bindings: BoundArguments
    ) -> str:
        """Determine the main input string for the given function `func`."""

        arguments = bindings.arguments

        message = arguments.get("message")
        if message is not None and not callable(message):
            content = _message_content(message)
            if content is not None:
                return content

        messages = _messages(arguments.get("messages"))
        if messages:
            content = _message_content(messages[-1])
            if content is not None:
                return content

        return core_app.App.main_input(self, func, sig, bindings)

    # App override:
    def main_output(
        self, func: Callable, sig: Signature, bindings: BoundArguments, ret: Any
    ) -> Optional[str]:
        """Determine the main output string for the given function `func`."""

        if isinstance(ret, ChatResult):
            summary = _chat_result_summary(ret)
            if summary is not None:
                return summary

        content = _message_content(ret)
        if content is not None:
            return content

        return core_app.App.main_output(self, func, sig, bindings, ret)

    def main_call(self, human: str) -> Optional[str]:
        """Ask the wrapped agent for a reply to a single message."""

        return _message_content(
            self.app.generate_reply(
                messages=[{"role": "user", "content": human}]
            )
        )

    async def main_acall(self, human: str) -> Optional[str]:
        """Ask the wrapped agent for a reply to a single message."""

        return _message_content(
            await self.app.a_generate_reply(
                messages=[{"role": "user", "content": human}]
            )
        )


TruAutoGen.model_rebuild()
