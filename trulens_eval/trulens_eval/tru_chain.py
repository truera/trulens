"""
# Langchain instrumentation and monitoring.
"""

from inspect import BoundArguments
from inspect import Signature
import logging
from pprint import PrettyPrinter
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

# import nest_asyncio # NOTE(piotrm): disabling for now, need more investigation
from pydantic import Field

from trulens_eval.app import App
from trulens_eval.instruments import Instrument
from trulens_eval.schema import Record
from trulens_eval.schema import Select
from trulens_eval.utils.asynchro import sync
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LANGCHAIN
from trulens_eval.utils.json import jsonify
from trulens_eval.utils.langchain import WithFeedbackFilterDocuments
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.pyschema import FunctionOrMethod
from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.serial import all_queries
from trulens_eval.utils.serial import Lens

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

with OptionalImports(messages=REQUIREMENT_LANGCHAIN):
    # langchain.agents.agent.AgentExecutor, # is langchain.chains.base.Chain
    # import langchain

    from langchain.agents.agent import BaseMultiActionAgent
    from langchain.agents.agent import BaseSingleActionAgent
    from langchain.chains.base import Chain
    from langchain.llms.base import BaseLLM
    from langchain.load.serializable import \
        Serializable  # this seems to be work in progress over at langchain
    from langchain.memory.chat_memory import BaseChatMemory
    from langchain.prompts.base import BasePromptTemplate
    from langchain.schema import BaseChatMessageHistory  # subclass of above
    from langchain.schema import BaseMemory  # no methods instrumented
    from langchain.schema import BaseRetriever
    from langchain.schema.document import Document
    from langchain.schema.language_model import BaseLanguageModel
    # langchain.adapters.openai.ChatCompletion, # no bases
    from langchain.tools.base import BaseTool
    from langchain_core.runnables.base import RunnableSerializable


class LangChainInstrument(Instrument):

    class Default:
        MODULES = {"langchain"}

        # Thunk because langchain is optional. TODO: Not anymore.
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
            WithFeedbackFilterDocuments
        }

        # Instrument only methods with these names and of these classes.
        METHODS = {
            "invoke":
                lambda o: isinstance(o, RunnableSerializable),
            "ainvoke":
                lambda o: isinstance(o, RunnableSerializable),
            "save_context":
                lambda o: isinstance(o, BaseMemory),
            "clear":
                lambda o: isinstance(o, BaseMemory),
            "_call":
                lambda o: isinstance(o, Chain),
            "__call__":
                lambda o: isinstance(o, Chain),
            "_acall":
                lambda o: isinstance(o, Chain),
            "acall":
                lambda o: isinstance(o, Chain),
            "_get_relevant_documents":
                lambda o: isinstance(o, (RunnableSerializable)),
            "_aget_relevant_documents":
                lambda o: isinstance(o, (RunnableSerializable)),
            "get_relevant_documents":
                lambda o: isinstance(o, (RunnableSerializable)),
            "aget_relevant_documents":
                lambda o: isinstance(o, (RunnableSerializable)),
            # "format_prompt": lambda o: isinstance(o, langchain.prompts.base.BasePromptTemplate),
            # "format": lambda o: isinstance(o, langchain.prompts.base.BasePromptTemplate),
            # the prompt calls might be too small to be interesting
            "plan":
                lambda o:
                isinstance(o, (BaseSingleActionAgent, BaseMultiActionAgent)),
            "aplan":
                lambda o:
                isinstance(o, (BaseSingleActionAgent, BaseMultiActionAgent)),
            "_arun":
                lambda o: isinstance(o, BaseTool),
            "_run":
                lambda o: isinstance(o, BaseTool),
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
    """Instantiates the Langchain Wrapper.
        
    Example:
        Langchain Code: [Langchain Quickstart](https://python.langchain.com/docs/get_started/quickstart)

        ```python
         # Code snippet taken from langchain 0.0.281 (API subject to change with new versions)
        from langchain.chains import LLMChain
        from langchain_community.llms import OpenAI
        from langchain.prompts import PromptTemplate
        from langchain.prompts.chat import ChatPromptTemplate
        from langchain.prompts.chat import HumanMessagePromptTemplate

        full_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=
                "Provide a helpful response with relevant background information for the following: {prompt}",
                input_variables=["prompt"],
            )
        )

        chat_prompt_template = ChatPromptTemplate.from_messages([full_prompt])

        llm = OpenAI(temperature=0.9, max_tokens=128)

        chain = LLMChain(llm=llm, prompt=chat_prompt_template, verbose=True)
        ```

        Trulens Eval Code:
        ```python
        
        from trulens_eval import TruChain
        # f_lang_match, f_qa_relevance, f_qs_relevance are feedback functions
        tru_recorder = TruChain(
            chain,
            app_id='Chain1_ChatApplication',
            feedbacks=[f_lang_match, f_qa_relevance, f_qs_relevance])
        )
        with tru_recorder as recording:
            chain(""What is langchain?")

        tru_record = recording.records[0]

        # To add record metadata 
        with tru_recorder as recording:
            recording.record_metadata="this is metadata for all records in this context that follow this line"
            chain("What is langchain?")
            recording.record_metadata="this is different metadata for all records in this context that follow this line"
            chain("Where do I download langchain?")
        ```

    See [Feedback Functions](https://www.trulens.org/trulens_eval/api/feedback/) for instantiating feedback functions.

    Args:
        app: A langchain application.

        **kwargs: Additional arguments to pass to [App][trulens_eval.app.App]
            and [AppDefinition][trulens_eval.app.AppDefinition]
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
        kwargs['app'] = app
        kwargs['root_class'] = Class.of_object(app)
        kwargs['instrument'] = LangChainInstrument(app=self)

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

            except Exception as e:
                pass

        if len(retrievers) == 0:
            raise ValueError("Cannot find any `BaseRetriever` in app.")

        if len(retrievers) > 1:
            raise ValueError(
                "Found more than one `BaseRetriever` in app:\n\t" + \
                ("\n\t".join(map(
                    lambda lr: f"{type(lr[1])} at {lr[0]}",
                    retrievers)))
            )

        return (
            Select.RecordCalls + retrievers[0][0]
        ).get_relevant_documents.rets

    def main_input(
        self, func: Callable, sig: Signature, bindings: BoundArguments
    ) -> str:
        """
        Determine the main input string for the given function `func` with
        signature `sig` if it is to be called with the given bindings
        `bindings`.
        """

        if 'inputs' in bindings.arguments \
            and safe_hasattr(self.app, "input_keys") \
            and safe_hasattr(self.app, "prep_inputs"):

            # langchain specific:
            ins = self.app.prep_inputs(bindings.arguments['inputs'])

            if len(self.app.input_keys) == 0:
                logger.warning(
                    "langchain app has no inputs. `main_input` will be `None`."
                )
                return None

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

        if isinstance(ret, Dict) and safe_hasattr(self.app, "output_keys"):
            # langchain specific:
            if self.app.output_keys[0] in ret:
                return ret[self.app.output_keys[0]]

        return App.main_output(self, func, sig, bindings, ret)

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


# from trulens_eval.utils import serial
# TruChain.model_rebuild()
