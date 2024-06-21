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
from pydantic import Field

from trulens_eval import app as mod_app
from trulens_eval.feedback import feedback
from trulens_eval.instruments import ClassFilter
from trulens_eval.instruments import Instrument
from trulens_eval.schema import feedback as mod_feedback_schema
from trulens_eval.tru_chain import LangChainInstrument
from trulens_eval.utils.containers import dict_set_with_multikey
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_RAILS
from trulens_eval.utils.json import jsonify
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.pyschema import FunctionOrMethod
from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.serial import Lens
from trulens_eval.utils.text import retab

logger = logging.getLogger(__name__)

with OptionalImports(messages=REQUIREMENT_RAILS) as opt:
    import nemoguardrails
    from nemoguardrails import LLMRails
    from nemoguardrails import RailsConfig
    from nemoguardrails.actions.action_dispatcher import ActionDispatcher
    from nemoguardrails.actions.actions import action
    from nemoguardrails.actions.actions import ActionResult
    from nemoguardrails.actions.llm.generation import LLMGenerationActions
    from nemoguardrails.kb.kb import KnowledgeBase
    from nemoguardrails.rails.llm.llmrails import LLMRails

opt.assert_installed(nemoguardrails)


class RailsActionSelect(mod_feedback_schema.Select):
    """Selector shorthands for _NeMo Guardrails_ apps when used for evaluating
    feedback in actions.
    
    These should not be used for feedback functions given to `TruRails` but
    instead for selectors in the `FeedbackActions` action invoked from with a
    rails app.
    """

    Action = Lens().action
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


class FeedbackActions():
    """Feedback action action for _NeMo Guardrails_ apps.
    
    See docstring of method `feedback`.
    """

    @staticmethod
    def register_feedback_functions(
        *args: Tuple[feedback.Feedback, ...], **kwargs: Dict[str,
                                                             feedback.Feedback]
    ):
        """Register one or more feedback functions to use in rails `feedback`
        action.
        
        All keyword arguments indicate the key as the keyword. All
        positional arguments use the feedback name as the key.
        """

        for name, feedback_instance in kwargs.items():
            if not isinstance(feedback_instance, feedback.Feedback):
                raise ValueError(
                    f"Invalid feedback function: {feedback_instance}; "
                    f"expected a Feedback class instance."
                )
            print(f"registered feedback function under name {name}")
            registered_feedback_functions[name] = feedback_instance

        for feedback_instance in args:
            if not isinstance(feedback_instance, feedback.Feedback):
                raise ValueError(
                    f"Invalid feedback function: {feedback_instance}; "
                    f"expected a Feedback class instance."
                )
            print(
                f"registered feedback function under name {feedback_instance.name}"
            )
            registered_feedback_functions[feedback_instance.name] = feedback

    @staticmethod
    def action_of_feedback(
        feedback_instance: feedback.Feedback,
        verbose: bool = False
    ) -> Callable:
        """Create a custom rails action for the given feedback function.
        
        Args:
            feedback_instance: A feedback function to register as an action.

            verbose: Print out info on invocation upon invocation.

        Returns:
            A custom action that will run the feedback function. The name is
                the same as the feedback function's name.
        """

        if not isinstance(feedback_instance, feedback.Feedback):
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
        selectors: Optional[Dict[str, Union[str, Lens]]] = None,
        verbose: bool = False
    ) -> ActionResult:
        """Run the specified feedback function from trulens_eval.
        
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
                + "\n\t".join(registered_feedback_functions.keys()) + "\n"
            )

        fname = feedback_function.name

        if selectors is None:
            raise ValueError(
                f"Need selectors for feedback function: {fname} "
                f"with signature {inspect.signature(feedback_function.imp)}"
            )

        selectors = {
            argname:
            (Lens.of_string(arglens) if isinstance(arglens, str) else arglens)
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
                    retab(
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
            LLMRails, KnowledgeBase, LLMGenerationActions, ActionDispatcher,
            FeedbackActions
        }.union(LangChainInstrument.Default.CLASSES())
        """Instrument only these classes."""

        METHODS: Dict[str, ClassFilter] = dict_set_with_multikey(
            dict(LangChainInstrument.Default.METHODS),  # copy
            {
                ("execute_action"):
                    ActionDispatcher,
                (
                    "generate", "generate_async", "stream_async", "generate_events", "generate_events_async", "_get_events_for_messages"
                ):
                    LLMRails,
                "search_relevant_chunks":
                    KnowledgeBase,
                (
                    "generate_user_intent", "generate_next_step", "generate_bot_message", "generate_value", "generate_intent_steps_message"
                ):
                    LLMGenerationActions,
                # TODO: Include feedback method in FeedbackActions, currently
                # bugged and will not be logged.
                "feedback":
                    FeedbackActions,
            }
        )
        """Instrument only methods with these names and of these classes."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            include_modules=RailsInstrument.Default.MODULES,
            include_classes=RailsInstrument.Default.CLASSES(),
            include_methods=RailsInstrument.Default.METHODS,
            *args,
            **kwargs
        )


class TruRails(mod_app.App):
    """Recorder for apps defined using _NeMo Guardrails_.

    Args:
        app: A _NeMo Guardrails_ application.
    """

    model_config: ClassVar[dict] = {'arbitrary_types_allowed': True}

    app: LLMRails

    root_callable: ClassVar[FunctionOrMethod] = Field(
        default_factory=lambda: FunctionOrMethod.of_callable(LLMRails.generate)
    )

    def __init__(self, app: LLMRails, **kwargs):
        # TruLlama specific:
        kwargs['app'] = app
        kwargs['root_class'] = Class.of_object(app)  # TODO: make class property
        kwargs['instrument'] = RailsInstrument(app=self)

        super().__init__(**kwargs)

    def main_output(
        self, func: Callable, sig: Signature, bindings: BoundArguments, ret: Any
    ) -> JSON:
        """
        Determine the main out string for the given function `func` with
        signature `sig` after it is called with the given `bindings` and has
        returned `ret`.
        """

        if isinstance(ret, dict):
            if "content" in ret:
                return ret['content']

        return jsonify(ret)

    def main_input(
        self, func: Callable, sig: Signature, bindings: BoundArguments
    ) -> JSON:
        """
        Determine the main input string for the given function `func` with
        signature `sig` after it is called with the given `bindings` and has
        returned `ret`.
        """

        if "messages" in bindings.arguments:
            messages = bindings.arguments['messages']
            if len(messages) == 1:
                message = messages[0]
                if "content" in message:
                    return message["content"]

        return jsonify(bindings.arguments)

    @classmethod
    def select_context(cls, app: Optional[LLMRails] = None) -> Lens:
        """
        Get the path to the context in the query output.
        """
        return mod_feedback_schema.Select.RecordCalls.kb.search_relevant_chunks.rets[:
                                                                                    ].body

    def __getattr__(self, name):
        if name == "__name__":
            return self.__class__.__name__  # Return the class name of TruRails
        elif safe_hasattr(self.app, name):
            return getattr(
                self.app, name
            )  # Delegate to the wrapped app if it has the attribute
        else:
            raise AttributeError(f"TruRails has no attribute named {name}")


import trulens_eval  # for App class annotations

TruRails.model_rebuild()
