"""
NeMo Guardrails instrumentation and monitoring.
"""

import inspect
from inspect import BoundArguments
from inspect import Signature
import logging
from pprint import pformat
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

from langchain_core.language_models.base import BaseLanguageModel
from nemoguardrails import LLMRails
from nemoguardrails import RailsConfig
from nemoguardrails.actions.action_dispatcher import ActionDispatcher
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.actions.actions import action
from nemoguardrails.actions.llm.generation import LLMGenerationActions
from nemoguardrails.kb.kb import KnowledgeBase
from pydantic import Field
from trulens.apps.langchain import LangChainInstrument
from trulens.core import app as core_app
from trulens.core.feedback import feedback as core_feedback
from trulens.core.instruments import Instrument
from trulens.core.instruments import InstrumentedMethod
from trulens.core.schema import select as select_schema
from trulens.core.utils import json as json_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import text as text_utils

logger = logging.getLogger(__name__)


class RailsActionSelect(select_schema.Select):
    """Selector shorthands for _NeMo Guardrails_ apps when used for evaluating
    feedback in actions.

    These should not be used for feedback functions given to `TruRails` but
    instead for selectors in the `FeedbackActions` action invoked from with a
    rails app.
    """

    Action = serial_utils.Lens().action
    """Selector for action call parameters."""

    Events = Action.events
    """Selector for events in action call parameters."""

    Context = Action.context
    """Selector for context in action call parameters.

    Warning:
        This is not the same "context" as in RAG triad. This is a parameter to rails
        actions that stores context of the rails app execution.
    """

    LLM = Action.llm
    """Selector for the language model in action call parameters."""

    Config = Action.config
    """Selector for the configuration in action call parameters."""

    RetrievalContexts = Context.relevant_chunks_sep
    """Selector for the retrieved contexts chunks returned from a KB search.

    Equivalent to `$relevant_chunks_sep` in colang."""

    UserMessage = Context.user_message
    """Selector for the user message.

    Equivalent to `$user_message` in colang."""

    BotMessage = Context.bot_message
    """Selector for the bot message.

    Equivalent to `$bot_message` in colang."""

    LastUserMessage = Context.last_user_message
    """Selector for the last user message.

    Equivalent to `$last_user_message` in colang."""

    LastBotMessage = Context.last_bot_message
    """Selector for the last bot message.

    Equivalent to `$last_bot_message` in colang."""


# NOTE(piotrm): Cannot have this inside FeedbackActions presently due to perhaps
# some closure-related issues with the @action decorator below.
registered_feedback_functions = {}


class FeedbackActions:
    """Feedback action action for _NeMo Guardrails_ apps.

    See docstring of method `feedback`.
    """

    @staticmethod
    def register_feedback_functions(
        *args: Tuple[core_feedback.Feedback, ...],
        **kwargs: Dict[str, core_feedback.Feedback],
    ):
        """Register one or more feedback functions to use in rails `feedback`
        action.

        All keyword arguments indicate the key as the keyword. All
        positional arguments use the feedback name as the key.
        """

        for name, feedback_instance in kwargs.items():
            if not isinstance(feedback_instance, core_feedback.Feedback):
                raise ValueError(
                    f"Invalid feedback function: {feedback_instance}; "
                    f"expected a Feedback class instance."
                )
            print(f"registered feedback function under name {name}")
            registered_feedback_functions[name] = feedback_instance

        for feedback_instance in args:
            if not isinstance(feedback_instance, core_feedback.Feedback):
                raise ValueError(
                    f"Invalid feedback function: {feedback_instance}; "
                    f"expected a Feedback class instance."
                )
            print(
                f"registered feedback function under name {feedback_instance.name}"
            )
            registered_feedback_functions[feedback_instance.name] = (
                feedback_instance
            )

    @staticmethod
    def action_of_feedback(
        feedback_instance: core_feedback.Feedback, verbose: bool = False
    ) -> Callable:
        """Create a custom rails action for the given feedback function.

        Args:
            feedback_instance: A feedback function to register as an action.

            verbose: Print out info on invocation upon invocation.

        Returns:
            A custom action that will run the feedback function. The name is
                the same as the feedback function's name.
        """

        if not isinstance(feedback_instance, core_feedback.Feedback):
            raise ValueError(
                f"Invalid feedback function: {feedback_instance}; "
                f"expected a Feedback class instance."
            )

        @action(name=feedback_instance.name)
        async def run_feedback(*args, **kwargs):
            if verbose:
                print(
                    f"Running feedback function {feedback_instance.name} with:"
                )
                print(f"  args = {args}")
                print(f"  kwargs = {kwargs}")

            res = feedback_instance.run(*args, **kwargs).result

            if verbose:
                print(f"  result = {res}")

            return res

        return run_feedback

    @action(name="feedback")
    @staticmethod
    async def feedback_action(
        # Default action arguments:
        events: Optional[List[Dict]] = None,
        context: Optional[Dict] = None,
        llm: Optional[BaseLanguageModel] = None,
        config: Optional[RailsConfig] = None,
        # Rest of arguments are specific to this action.
        function: Optional[str] = None,
        selectors: Optional[Dict[str, Union[str, serial_utils.Lens]]] = None,
        verbose: bool = False,
    ) -> ActionResult:
        """Run the specified feedback function from trulens.

        To use this action, it needs to be registered with your rails app and
        feedback functions themselves need to be registered with this function.
        The name under which this action is registered for rails is `feedback`.

        Usage:
            ```python
            rails: LLMRails = ... # your app
            language_match: Feedback = Feedback(...) # your feedback function

            # First we register some feedback functions with the custom action:
            FeedbackAction.register_feedback_functions(language_match)

            # Can also use kwargs expansion from dict like produced by rag_triad:
            # FeedbackAction.register_feedback_functions(**rag_triad(...))

            # Then the feedback method needs to be registered with the rails app:
            rails.register_action(FeedbackAction.feedback)
            ```

        Args:
            events: See [Action
                parameters](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/docs/user_guides/python-api.md#special-parameters).

            context: See [Action
                parameters](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/docs/user_guides/python-api.md#special-parameters).

            llm: See [Action
                parameters](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/docs/user_guides/python-api.md#special-parameters).

            config: See [Action
                parameters](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/docs/user_guides/python-api.md#special-parameters).

            function: Name of the feedback function to run.

            selectors: Selectors for the function. Can be provided either as
                strings to be parsed into lenses or lenses themselves.

            verbose: Print the values of the selectors before running feedback
                and print the result after running feedback.

        Returns:
            ActionResult: An action result containing the result of the feedback.

        Example:
            ```colang
            define subflow check language match
                $result = execute feedback(\\
                    function="language_match",\\
                    selectors={\\
                    "text1":"action.context.last_user_message",\\
                    "text2":"action.context.bot_message"\\
                    }\\
                )
                if $result < 0.8
                    bot inform language mismatch
                    stop
            ```
        """

        feedback_function = registered_feedback_functions.get(function)

        if feedback_function is None:
            raise ValueError(
                f"Invalid feedback function: {function}; "
                f"there is/are {len(registered_feedback_functions)} registered function(s):\n\t"
                + "\n\t".join(registered_feedback_functions.keys())
                + "\n"
            )

        fname = feedback_function.name

        if selectors is None:
            raise ValueError(
                f"Need selectors for feedback function: {fname} "
                f"with signature {inspect.signature(feedback_function.imp)}"
            )

        selectors = {
            argname: (
                serial_utils.Lens.of_string(arglens)
                if isinstance(arglens, str)
                else arglens
            )
            for argname, arglens in selectors.items()
        }

        feedback_function = feedback_function.on(**selectors)

        source_data = dict(
            action=dict(events=events, context=context, llm=llm, config=config)
        )

        if verbose:
            print(fname)
            for argname, lens in feedback_function.selectors.items():
                print(f"  {argname} = ", end=None)
                # use pretty print for the potentially big thing here:
                print(
                    text_utils.retab(
                        tab="    ", s=pformat(lens.get_sole_item(source_data))
                    )
                )

        context_updates = {}

        try:
            result = feedback_function.run(source_data=source_data)
            context_updates["result"] = result.result

            if verbose:
                print(f"  {fname} result = {result.result}")

        except Exception:
            context_updates["result"] = None

            return ActionResult(
                return_value=context_updates["result"],
                context_updates=context_updates,
            )

        return ActionResult(
            return_value=context_updates["result"],
            context_updates=context_updates,
        )


class RailsInstrument(Instrument):
    """Instrumentation specification for _NeMo Guardrails_ apps."""

    class Default:
        """Default instrumentation specification."""

        MODULES = {"nemoguardrails"}.union(LangChainInstrument.Default.MODULES)
        """Modules to instrument by name prefix.

        Note that _NeMo Guardrails_ uses _LangChain_ internally for some things.
        """

        CLASSES = lambda: {
            LLMRails,
            KnowledgeBase,
            LLMGenerationActions,
            ActionDispatcher,
            FeedbackActions,
        }.union(LangChainInstrument.Default.CLASSES())
        """Instrument only these classes."""

        METHODS: List[InstrumentedMethod] = (
            LangChainInstrument.Default.METHODS
            + [
                InstrumentedMethod("execute_action", ActionDispatcher),
                InstrumentedMethod("generate", LLMRails),
                InstrumentedMethod("generate_async", LLMRails),
                InstrumentedMethod("stream_async", LLMRails),
                InstrumentedMethod("generate_events", LLMRails),
                InstrumentedMethod("generate_events_async", LLMRails),
                InstrumentedMethod("_get_events_for_messages", LLMRails),
                InstrumentedMethod("search_relevant_chunks", KnowledgeBase),
                InstrumentedMethod(
                    "generate_user_intent", LLMGenerationActions
                ),
                InstrumentedMethod("generate_next_step", LLMGenerationActions),
                InstrumentedMethod(
                    "generate_bot_message", LLMGenerationActions
                ),
                InstrumentedMethod("generate_value", LLMGenerationActions),
                InstrumentedMethod(
                    "generate_intent_steps_message", LLMGenerationActions
                ),
                InstrumentedMethod("feedback", FeedbackActions),
            ]
        )
        """Instrument only methods with these names and of these classes."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            include_modules=RailsInstrument.Default.MODULES,
            include_classes=RailsInstrument.Default.CLASSES(),
            include_methods=RailsInstrument.Default.METHODS,
            *args,
            **kwargs,
        )


class TruRails(core_app.App):
    """Recorder for apps defined using _NeMo Guardrails_.

    Args:
        app: A _NeMo Guardrails_ application.
    """

    app: LLMRails

    # TODEP
    root_callable: ClassVar[pyschema_utils.FunctionOrMethod] = Field(None)

    def __init__(self, app: LLMRails, **kwargs):
        # TruLlama specific:
        kwargs["app"] = app
        kwargs["root_class"] = pyschema_utils.Class.of_object(
            app
        )  # TODO: make class property
        kwargs["instrument"] = RailsInstrument(app=self)

        super().__init__(**kwargs)

    def main_output(
        self, func: Callable, sig: Signature, bindings: BoundArguments, ret: Any
    ) -> serial_utils.JSON:
        """
        Determine the main out string for the given function `func` with
        signature `sig` after it is called with the given `bindings` and has
        returned `ret`.
        """

        if isinstance(ret, dict):
            if "content" in ret:
                return ret["content"]

        return json_utils.jsonify(ret)

    def main_input(
        self, func: Callable, sig: Signature, bindings: BoundArguments
    ) -> serial_utils.JSON:
        """
        Determine the main input string for the given function `func` with
        signature `sig` after it is called with the given `bindings` and has
        returned `ret`.
        """

        if "messages" in bindings.arguments:
            messages = bindings.arguments["messages"]
            if len(messages) == 1:
                message = messages[0]
                if "content" in message:
                    return message["content"]

        return json_utils.jsonify(bindings.arguments)

    @classmethod
    def select_context(
        cls, app: Optional[LLMRails] = None
    ) -> serial_utils.Lens:
        """
        Get the path to the context in the query output.
        """
        return select_schema.Select.RecordCalls.kb.search_relevant_chunks.rets[
            :
        ].body

    def __getattr__(self, name):
        if name == "__name__":
            return self.__class__.__name__  # Return the class name of TruRails
        elif python_utils.safe_hasattr(self.app, name):
            return getattr(
                self.app, name
            )  # Delegate to the wrapped app if it has the attribute
        else:
            raise AttributeError(f"TruRails has no attribute named {name}")


TruRails.model_rebuild()
