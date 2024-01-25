"""
# NEMO Guardrails instrumentation and monitoring. 
"""

from inspect import BoundArguments
from inspect import Signature
import logging
from pprint import PrettyPrinter
from typing import Any, Callable, ClassVar, Optional

from pydantic import Field

from trulens_eval.app import App
from trulens_eval.instruments import Instrument
from trulens_eval.schema import Select
from trulens_eval.utils.containers import dict_set_with_multikey
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_RAILS
from trulens_eval.utils.json import jsonify
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.pyschema import FunctionOrMethod
from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.serial import Lens

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

with OptionalImports(messages=REQUIREMENT_RAILS):
    import nemoguardrails
    from nemoguardrails.actions.llm.generation import LLMGenerationActions
    from nemoguardrails.flows.runtime import Runtime
    from nemoguardrails.kb.kb import KnowledgeBase
    from nemoguardrails.rails.llm.llmrails import LLMRails
    from nemoguardrails.actions.action_dispatcher import ActionDispatcher
    
OptionalImports(messages=REQUIREMENT_RAILS).assert_installed(nemoguardrails)

from trulens_eval.tru_chain import LangChainInstrument


class RailsInstrument(Instrument):

    class Default:
        MODULES = {"nemoguardrails"}.union(
            LangChainInstrument.Default.MODULES
        )  # NOTE: nemo uses langchain internally for some things

        # Putting these inside thunk as llama_index is optional.
        CLASSES = lambda: {
            LLMRails, KnowledgeBase, LLMGenerationActions, Runtime, ActionDispatcher
        }.union(LangChainInstrument.Default.CLASSES())

        # Instrument only methods with these names and of these classes. Ok to
        # include llama_index inside methods.
        METHODS = dict_set_with_multikey(
            dict(LangChainInstrument.Default.METHODS), # copy
            {
                ("execute_action"): lambda o: isinstance(o, ActionDispatcher),
                (
                    "generate", "generate_async",
                    "stream_async",
                    "generate_events", "generate_events_async", "_get_events_for_messages"
                ): lambda o: isinstance(o, LLMRails),
                "search_relevant_chunks": lambda o: isinstance(o, KnowledgeBase),
                (
                    "generate_user_intent",
                    "generate_next_step",
                    "generate_bot_message",
                    "generate_value",
                    "generate_intent_steps_message"
                ): lambda o: isinstance(o, LLMGenerationActions),
                (
                    "generate_events",
                    "compute_next_steps"
                ): lambda o: isinstance(o, Runtime)
            }
        )

    def __init__(self, *args, **kwargs):
        super().__init__(
            include_modules=RailsInstrument.Default.MODULES,
            include_classes=RailsInstrument.Default.CLASSES(),
            include_methods=RailsInstrument.Default.METHODS,
            *args,
            **kwargs
        )


class TruRails(App):
    """
    Recorder for apps defined using NEMO guardrails.

        Args:
            app -- A nemo guardrails application.
    """

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

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
    def select_context(
        cls,
        app: Optional[LLMRails] = None
    ) -> Lens:
        """
        Get the path to the context in the query output.
        """
        return Select.RecordCalls.kb.search_relevant_chunks.rets[:].body

    def __getattr__(self, __name: str) -> Any:
        # A message for cases where a user calls something that the wrapped
        # app has but we do not wrap yet.

        if safe_hasattr(self.app, __name):
            return RuntimeError(
                f"TruRails has no attribute {__name} but the wrapped app ({type(self.app)}) does. ",
                f"If you are calling a {type(self.app)} method, retrieve it from that app instead of from `TruRails`. "
            )
        else:
            raise RuntimeError(f"TruRails has no attribute named {__name}.")

TruRails.model_rebuild()
