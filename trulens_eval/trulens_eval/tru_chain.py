"""
# Langchain instrumentation and monitoring.
"""

from inspect import BoundArguments
from inspect import Signature
import logging
from pprint import PrettyPrinter
from typing import Any, Callable, ClassVar, Dict, List, Tuple

# import nest_asyncio # NOTE(piotrm): disabling for now, need more investigation
from pydantic import Field

from trulens_eval.app import App
from trulens_eval.instruments import Instrument
from trulens_eval.schema import Record
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LANGCHAIN
from trulens_eval.utils.langchain import WithFeedbackFilterDocuments
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.pyschema import FunctionOrMethod

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

with OptionalImports(message=REQUIREMENT_LANGCHAIN):
    import langchain
    from langchain.chains.base import Chain


class LangChainInstrument(Instrument):

    class Default:
        MODULES = {"langchain."}

        # Thunk because langchain is optional.
        CLASSES = lambda: {
            langchain.chains.base.Chain,
            langchain.vectorstores.base.BaseRetriever,
            langchain.schema.BaseRetriever,
            langchain.llms.base.BaseLLM,
            langchain.prompts.base.BasePromptTemplate,
            langchain.schema.BaseMemory,  # no methods instrumented
            langchain.schema.BaseChatMessageHistory,  # subclass of above
            # langchain.agents.agent.AgentExecutor, # is langchain.chains.base.Chain
            WithFeedbackFilterDocuments
        }

        # Instrument only methods with these names and of these classes.
        METHODS = {
            "_call": lambda o: isinstance(o, langchain.chains.base.Chain),
            "__call__": lambda o: isinstance(o, langchain.chains.base.Chain),
            "_acall": lambda o: isinstance(o, langchain.chains.base.Chain),
            "acall": lambda o: isinstance(o, langchain.chains.base.Chain),
            "_get_relevant_documents":
                lambda o: True,  # VectorStoreRetriever, langchain >= 0.230
        }

    def __init__(self, *args, **kwargs):
        super().__init__(
            include_modules=LangChainInstrument.Default.MODULES,
            include_classes=LangChainInstrument.Default.CLASSES(),
            include_methods=LangChainInstrument.Default.METHODS,
            *args,
            **kwargs
        )


class TruChain(App):
    """
    Wrap a langchain Chain to capture its configuration and evaluation steps. 
    """

    app: Chain

    # TODO: what if _acall is being used instead?
    root_callable: ClassVar[FunctionOrMethod] = Field(
        default_factory=lambda: FunctionOrMethod.of_callable(TruChain._call),
        const=True
    )

    # Normally pydantic does not like positional args but chain here is
    # important enough to make an exception.
    def __init__(self, app: Chain, **kwargs):
        """
        Wrap a langchain chain for monitoring.

        Arguments:
        - app: Chain -- the chain to wrap.
        - More args in App
        - More args in AppDefinition
        - More args in WithClassInfo
        """

        super().update_forward_refs()

        # TruChain specific:
        kwargs['app'] = app
        kwargs['root_class'] = Class.of_object(app)
        kwargs['instrument'] = LangChainInstrument(
            root_methods=set([TruChain.with_record, TruChain.awith_record]),
            callbacks=self
        )

        super().__init__(**kwargs)

        self.post_init()

    # TODEP
    # Chain requirement
    @property
    def _chain_type(self):
        return "TruChain"

    # TODEP
    # Chain requirement
    @property
    def input_keys(self) -> List[str]:
        return self.app.input_keys

    # TODEP
    # Chain requirement
    @property
    def output_keys(self) -> List[str]:
        return self.app.output_keys

    def main_input(
        self, func: Callable, sig: Signature, bindings: BoundArguments
    ) -> str:
        """
        Determine the main input string for the given function `func` with
        signature `sig` if it is to be called with the given bindings
        `bindings`.
        """

        if 'inputs' in bindings.arguments:
            # langchain specific:
            ins = self.app.prep_inputs(bindings.arguments['inputs'])

            return ins[self.app.input_keys[0]]

        return App.main_input(self, func, sig, bindings)

    def main_output(
        self, func: Callable, sig: Signature, bindings: BoundArguments, ret: Any
    ) -> str:
        """
        Determine the main out string for the given function `func` with
        signature `sig` after it is called with the given `bindings` and has
        returned `ret`.
        """

        if isinstance(ret, Dict):
            # langchain specific:
            if self.app.output_keys[0] in ret:
                return ret[self.app.output_keys[0]]

        return App.main_output(self, func, sig, bindings, ret)

    def __getattr__(self, __name: str) -> Any:
        # A message for cases where a user calls something that the wrapped
        # chain has but we do not wrap yet.

        if hasattr(self.app, __name):
            return RuntimeError(
                f"TruChain has no attribute {__name} but the wrapped app ({type(self.app)}) does. ",
                f"If you are calling a {type(self.app)} method, retrieve it from that app instead of from `TruChain`. "
                f"TruChain presently only wraps Chain.__call__, Chain._call, and Chain._acall ."
            )
        else:
            raise RuntimeError(f"TruChain has no attribute named {__name}.")

    # NOTE: Input signature compatible with langchain.chains.base.Chain.acall
    # TODEP
    async def acall_with_record(self, *args, **kwargs) -> Tuple[Any, Record]:
        """
        Run the chain acall method and also return a record metadata object.
        """
        return await self.awith_record(self.app.acall, *args, **kwargs)

    # NOTE: Input signature compatible with langchain.chains.base.Chain.__call__
    # TODEP
    def call_with_record(self, *args, **kwargs) -> Tuple[Any, Record]:
        """
        Run the chain call method and also return a record metadata object.
        """
        return self.with_record(self.app.__call__, *args, **kwargs)

    # TODEP
    # Mimics Chain
    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Wrapped call to self.app._call with instrumentation. If you need to
        get the record, use `call_with_record` instead. 
        """

        return self._call(*args, **kwargs)

    # TODEP
    # Chain requirement
    def _call(self, *args, **kwargs) -> Any:
        ret, _ = self.call_with_record(*args, **kwargs)

        return ret

    # TODEP
    # Optional Chain requirement
    async def _acall(self, *args, **kwargs) -> Any:
        ret, _ = await self.acall_with_record(*args, **kwargs)

        return ret
