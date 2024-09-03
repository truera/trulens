"""
# LangChain app instrumentation.
"""

from inspect import BoundArguments
from inspect import Signature
import logging
from pprint import PrettyPrinter
from typing import Any, Callable, ClassVar, Dict, Optional

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.runnables.base import RunnableSerializable

# import nest_asyncio # NOTE(piotrm): disabling for now, need more investigation
from pydantic import Field
from trulens.apps.langchain.guardrails import WithFeedbackFilterDocuments
from trulens.core import app as mod_app
from trulens.core.instruments import ClassFilter
from trulens.core.instruments import Instrument
from trulens.core.schema.select import Select
from trulens.core.utils.containers import dict_set_with_multikey
from trulens.core.utils.json import jsonify
from trulens.core.utils.pyschema import Class
from trulens.core.utils.pyschema import FunctionOrMethod
from trulens.core.utils.python import safe_hasattr
from trulens.core.utils.serial import Lens
from trulens.core.utils.serial import all_queries

from langchain.agents.agent import BaseMultiActionAgent
from langchain.agents.agent import BaseSingleActionAgent
from langchain.chains.base import Chain
from langchain.load.serializable import Serializable

# this seems to be work in progress over at langchain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts.base import BasePromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import BaseChatMessageHistory  # subclass of above
from langchain.schema import BaseMemory  # no methods instrumented
from langchain.schema import BaseRetriever
from langchain.schema.document import Document

# langchain.adapters.openai.ChatCompletion, # no bases
from langchain.tools.base import BaseTool

logger = logging.getLogger(__name__)

pp = PrettyPrinter()


class LangChainInstrument(Instrument):
    """Instrumentation for LangChain apps."""

    class Default:
        """Instrumentation specification for LangChain apps."""

        MODULES = {"langchain"}
        """Filter for module name prefix for modules to be instrumented."""

        CLASSES = lambda: {
            RunnableSerializable,
            Serializable,
            Document,
            Chain,
            BaseRetriever,
            BaseLLM,
            BasePromptTemplate,
            BaseMemory,  # no methods instrumented
            BaseChatMemory,  # no methods instrumented
            BaseChatMessageHistory,  # subclass of above
            # langchain.agents.agent.AgentExecutor, # is langchain.chains.base.Chain
            BaseSingleActionAgent,
            BaseMultiActionAgent,
            BaseLanguageModel,
            # langchain.load.serializable.Serializable, # this seems to be work in progress over at langchain
            # langchain.adapters.openai.ChatCompletion, # no bases
            BaseTool,
            WithFeedbackFilterDocuments,
        }
        """Filter for classes to be instrumented."""

        # Instrument only methods with these names and of these classes.
        METHODS: Dict[str, ClassFilter] = dict_set_with_multikey(
            {},
            {
                ("invoke", "ainvoke"): RunnableSerializable,
                ("save_context", "clear"): BaseMemory,
                ("run", "arun", "_call", "__call__", "_acall", "acall"): Chain,
                (
                    "_get_relevant_documents",
                    "get_relevant_documents",
                    "aget_relevant_documents",
                    "_aget_relevant_documents",
                    "invoke",
                    "ainvoke",
                ): RunnableSerializable,
                # "format_prompt": lambda o: isinstance(o, langchain.prompts.base.BasePromptTemplate),
                # "format": lambda o: isinstance(o, langchain.prompts.base.BasePromptTemplate),
                # the prompt calls might be too small to be interesting
                ("plan", "aplan"): (
                    BaseSingleActionAgent,
                    BaseMultiActionAgent,
                ),
                ("_arun", "_run"): BaseTool,
            },
        )
        """Methods to be instrumented.

        Key is method name and value is filter for objects that need those
        methods instrumented"""

    def __init__(self, *args, **kwargs):
        super().__init__(
            include_modules=LangChainInstrument.Default.MODULES,
            include_classes=LangChainInstrument.Default.CLASSES(),
            include_methods=LangChainInstrument.Default.METHODS,
            *args,
            **kwargs,
        )


class TruChain(mod_app.App):
    """Recorder for _LangChain_ applications.

    This recorder is designed for LangChain apps, providing a way to instrument,
    log, and evaluate their behavior.

    Example: "Creating a LangChain RAG application"

        Consider an example LangChain RAG application. For the complete code
        example, see [LangChain
        Quickstart](https://www.trulens.org/trulens/getting_started/quickstarts/langchain_quickstart/).

        ```python
        from langchain import hub
        from langchain.chat_models import ChatOpenAI
        from langchain.schema import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        retriever = vectorstore.as_retriever()

        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        ```

    Feedback functions can utilize the specific context produced by the
    application's retriever. This is achieved using the `select_context` method,
    which then can be used by a feedback selector, such as `on(context)`.

    Example: "Defining a feedback function"

        ```python
        from trulens.providers.openai import OpenAI
        from trulens.core import Feedback
        import numpy as np

        # Select context to be used in feedback.
        from trulens.apps.langchain import TruChain
        context = TruChain.select_context(rag_chain)


        # Use feedback
        f_context_relevance = (
            Feedback(provider.context_relevance_with_context_reasons)
            .on_input()
            .on(context)  # Refers to context defined from `select_context`
            .aggregate(np.mean)
        )
        ```

    The application can be wrapped in a `TruChain` recorder to provide logging
    and evaluation upon the application's use.

    Example: "Using the `TruChain` recorder"

        ```python
        from trulens.apps.langchain import TruChain

        # Wrap application
        tru_recorder = TruChain(
            chain,
            app_name="ChatApplication",
            app_version="chain_v1",
            feedbacks=[f_context_relevance]
        )

        # Record application runs
        with tru_recorder as recording:
            chain("What is langchain?")
        ```

    Further information about LangChain apps can be found on the [LangChain
    Documentation](https://python.langchain.com/docs/) page.

    Args:
        app: A LangChain application.

        **kwargs: Additional arguments to pass to [App][trulens.core.app.App]
            and [AppDefinition][trulens.core.schema.app.AppDefinition].
    """

    app: Any  # Chain
    """The langchain app to be instrumented."""

    # TODO: what if _acall is being used instead?
    root_callable: ClassVar[FunctionOrMethod] = Field(
        default_factory=lambda: FunctionOrMethod.of_callable(TruChain._call)
    )
    """The root callable of the wrapped app."""

    # Normally pydantic does not like positional args but chain here is
    # important enough to make an exception.
    def __init__(self, app: Chain, **kwargs: dict):
        # TruChain specific:
        kwargs["app"] = app
        kwargs["root_class"] = Class.of_object(app)
        kwargs["instrument"] = LangChainInstrument(app=self)

        super().__init__(**kwargs)

    @classmethod
    def select_context(cls, app: Optional[Chain] = None) -> Lens:
        """Get the path to the context in the query output."""

        if app is None:
            raise ValueError(
                "langchain app/chain is required to determine context for langchain apps. "
                "Pass it in as the `app` argument"
            )

        retrievers = []

        app_json = jsonify(app)
        for lens in all_queries(app_json):
            try:
                comp = lens.get_sole_item(app)
                if isinstance(comp, BaseRetriever):
                    retrievers.append((lens, comp))

            except Exception:
                pass

        if len(retrievers) == 0:
            raise ValueError("Cannot find any `BaseRetriever` in app.")

        if len(retrievers) > 1:
            if isinstance(retrievers[0][1], MultiQueryRetriever):
                pass
            else:
                raise ValueError(
                    "Found more than one `BaseRetriever` in app:\n\t"
                    + (
                        "\n\t".join(
                            map(
                                lambda lr: f"{type(lr[1])} at {lr[0]}",
                                retrievers,
                            )
                        )
                    )
                )

        retriever = Select.RecordCalls + retrievers[0][0]
        if hasattr(retriever, "invoke"):
            return retriever.invoke.rets[:].page_content
        if hasattr(retriever, "_get_relevant_documents"):
            return retriever._get_relevant_documents.rets[:].page_content

    def main_input(
        self, func: Callable, sig: Signature, bindings: BoundArguments
    ) -> str:  # might have to relax to JSON output
        """
        Determine the main input string for the given function `func` with
        signature `sig` if it is to be called with the given bindings
        `bindings`.
        """

        if "input" in bindings.arguments:
            temp = bindings.arguments["input"]
            if isinstance(temp, str):
                return temp

            if isinstance(temp, dict):
                vals = list(temp.values())
            elif isinstance(temp, list):
                vals = temp

            if len(vals) == 0:
                return None

            if len(vals) == 1:
                return vals[0]

            if len(vals) > 1:
                return vals[0]

        if (
            "inputs" in bindings.arguments
            and safe_hasattr(self.app, "input_keys")
            and safe_hasattr(self.app, "prep_inputs")
        ):
            # langchain specific:
            ins = self.app.prep_inputs(bindings.arguments["inputs"])

            if len(self.app.input_keys) == 0:
                logger.warning(
                    "langchain app has no `input_keys`. `main_input` might not be detected."
                )
                return super().main_input(func, sig, bindings)

            return ins[self.app.input_keys[0]]

        return mod_app.App.main_input(self, func, sig, bindings)

    def main_output(
        self, func: Callable, sig: Signature, bindings: BoundArguments, ret: Any
    ) -> str:
        """
        Determine the main out string for the given function `func` with
        signature `sig` after it is called with the given `bindings` and has
        returned `ret`.
        """

        if isinstance(ret, Dict) and safe_hasattr(self.app, "output_keys"):
            # langchain specific:
            if len(self.app.output_keys) == 0:
                logger.warning(
                    "langchain app has no `output_keys`. `main_output` might not be detected."
                )
                return super().main_output(func, sig, bindings, ret)

            if self.app.output_keys[0] in ret:
                return ret[self.app.output_keys[0]]

        return mod_app.App.main_output(self, func, sig, bindings, ret)

    def main_call(self, human: str):
        # If available, a single text to a single text invocation of this app.

        if safe_hasattr(self.app, "output_keys"):
            out_key = self.app.output_keys[0]
            return self.app(human)[out_key]
        else:
            logger.warning("Unsure what the main output string may be.")
            return str(self.app(human))

    async def main_acall(self, human: str):
        # If available, a single text to a single text invocation of this app.

        out = await self._acall(human)

        if safe_hasattr(self.app, "output_keys"):
            out_key = self.app.output_keys[0]
            return out[out_key]
        else:
            logger.warning("Unsure what the main output string may be.")
            return str(out)

    # NOTE: Input signature compatible with langchain.chains.base.Chain.acall
    # TOREMOVE
    async def acall_with_record(self, *args, **kwargs) -> None:
        """
        DEPRECATED: Run the chain acall method and also return a record metadata object.
        """

        self._throw_dep_message(method="acall", is_async=True, with_record=True)

    # NOTE: Input signature compatible with langchain.chains.base.Chain.__call__
    # TOREMOVE
    def call_with_record(self, *args, **kwargs) -> None:
        """
        DEPRECATED: Run the chain call method and also return a record metadata object.
        """

        self._throw_dep_message(
            method="__call__", is_async=False, with_record=True
        )

    # TOREMOVE
    # Mimics Chain
    def __call__(self, *args, **kwargs) -> None:
        """
        DEPRECATED: Wrapped call to self.app._call with instrumentation. If you
        need to get the record, use `call_with_record` instead.
        """
        self._throw_dep_message(
            method="__call__", is_async=False, with_record=False
        )

    # TOREMOVE
    # Chain requirement
    def _call(self, *args, **kwargs) -> None:
        self._throw_dep_message(
            method="_call", is_async=False, with_record=False
        )

    # TOREMOVE
    # Optional Chain requirement
    async def _acall(self, *args, **kwargs) -> None:
        self._throw_dep_message(
            method="_acall", is_async=True, with_record=False
        )


TruChain.model_rebuild()
