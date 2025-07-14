from __future__ import annotations

from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
import contextvars
import datetime
import inspect
from inspect import BoundArguments
from inspect import Signature
import json
import logging
import threading
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import weakref

import pandas as pd
import pydantic
from trulens.core import experimental as core_experimental
from trulens.core import instruments as core_instruments
from trulens.core import session as core_session
from trulens.core._utils import optional as optional_utils
from trulens.core._utils.pycompat import Future  # import standard exception
from trulens.core.database import base as core_db
from trulens.core.database import connector as core_connector
from trulens.core.feedback import endpoint as core_endpoint
from trulens.core.feedback import feedback as core_feedback
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.core.run import Run
from trulens.core.run import RunConfig
from trulens.core.run import validate_dataset_spec
from trulens.core.schema import app as app_schema
from trulens.core.schema import base as base_schema
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import record as record_schema
from trulens.core.schema import select as select_schema
from trulens.core.session import TruSession
from trulens.core.utils import asynchro as asynchro_utils
from trulens.core.utils import constants as constant_utils
from trulens.core.utils import containers as container_utils
from trulens.core.utils import deprecation as deprecation_utils
from trulens.core.utils import evaluator as evaluator_utils
from trulens.core.utils import imports as import_utils
from trulens.core.utils import json as json_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import signature as signature_utils
from trulens.core.utils import threading as threading_utils
from trulens.feedback.computer import compute_feedback_by_span_group
from trulens.otel.semconv.constants import (
    TRULENS_APP_SPECIFIC_INSTRUMENT_WRAPPER_FLAG,
)
from trulens.otel.semconv.constants import TRULENS_INSTRUMENT_WRAPPER_FLAG
from trulens.otel.semconv.constants import (
    TRULENS_RECORD_ROOT_INSTRUMENT_WRAPPER_FLAG,
)

logger = logging.getLogger(__name__)

# App component.
COMPONENT = Any

A = TypeVar("A")
T = TypeVar("T")

# Message produced when an attribute is looked up from our App but is actually
# an attribute of the enclosed app.
ATTRIBUTE_ERROR_MESSAGE = """
{class_name} has no attribute `{attribute_name}` but the wrapped app {app_class_name} does. If
you are calling a {app_class_name} method, retrieve it from that app instead of from
{class_name}. If you need to record your app's behavior, use {class_name} as a context
manager as in this Example:
```python
    app: {app_class_name} = ...  # your app
    truapp: {class_name} = {class_name}(app, ...)  # the truera recorder

    with truapp as recorder:
      result = app.{attribute_name}(...)

    record: Record = recorder.get() # get the record of the invocation if needed
```
"""

_component_impls: Dict[str, Type[ComponentView]] = {}


class ComponentViewMeta(ABCMeta):
    def __init__(
        cls,
        classname: str,
        bases: Tuple[Type[ComponentView]],
        dict_: Dict[str, Any],
    ):
        newtype = type.__init__(cls, classname, bases, dict_)

        if hasattr(cls, "component_of_json"):
            # Only register the higher-level classes that can enumerate this
            # subclasses. Otherwise an infinite loop will result in the of_json
            # logic below.

            _component_impls[classname] = cls  # type: ignore[assignment]
        return newtype


class ComponentView(ABC, metaclass=ComponentViewMeta):
    """
    Views of common app component types for sorting them and displaying them in
    some unified manner in the UI. Operates on components serialized into json
    dicts representing various components, not the components themselves.
    """

    def __init__(self, json: serial_utils.JSON):
        self.json = json
        self.cls = pyschema_utils.Class.of_class_info(json)

    @classmethod
    def of_json(cls, json: serial_utils.JSON) -> "ComponentView":
        """Sort the given json into the appropriate component view type."""

        cls_obj = pyschema_utils.Class.of_class_info(json)

        for _, view in _component_impls.items():
            # NOTE: includes prompt, llm, tool, agent, memory, other which may be overridden
            if view.class_is(cls_obj):
                return view.of_json(json)

        raise TypeError(f"Unhandled component type with class {cls_obj}")

    @staticmethod
    @abstractmethod
    def class_is(cls_obj: pyschema_utils.Class) -> bool:
        """Determine whether the given class representation `cls` is of the type to
        be viewed as this component type."""
        pass

    def unsorted_parameters(
        self, skip: Set[str]
    ) -> Dict[str, serial_utils.JSON_BASES_T]:
        """All basic parameters not organized by other accessors."""

        ret = {}

        for k, v in self.json.items():
            if k not in skip and isinstance(v, serial_utils.JSON_BASES):
                ret[k] = v

        return ret

    @staticmethod
    def innermost_base(
        bases: Optional[Sequence[pyschema_utils.Class]] = None,
        among_modules=set(["langchain", "llama_index", "trulens"]),
    ) -> Optional[str]:
        """
        Given a sequence of classes, return the first one which comes from one
        of the `among_modules`. You can use this to determine where ultimately
        the encoded class comes from in terms of langchain, llama_index, or
        trulens even in cases they extend each other's classes. Returns
        None if no module from `among_modules` is named in `bases`.
        """
        if bases is None:
            return None

        for base in bases:
            if "." in base.module.module_name:
                root_module = base.module.module_name.split(".")[0]
            else:
                root_module = base.module.module_name

            if root_module in among_modules:
                return root_module

        return None


class TrulensComponent(ComponentView):
    """Components provided in trulens."""

    @staticmethod
    def class_is(cls_obj: pyschema_utils.Class) -> bool:
        if ComponentView.innermost_base(cls_obj.bases) == "trulens":
            return True

        # if any(base.module.module_name.startswith("trulens.") for base in cls.bases):
        #    return True

        return False

    @staticmethod
    def of_json(json: serial_utils.JSON) -> "TrulensComponent":
        # NOTE: This import is here to avoid circular imports.
        from trulens.core.utils.trulens import component_of_json

        return component_of_json(json)


class Prompt(ComponentView):
    # langchain.prompts.base.BasePromptTemplate
    # llama_index.prompts.base.Prompt

    @property
    @abstractmethod
    def template(self) -> str:
        pass


class LLM(ComponentView):
    # langchain.llms.base.BaseLLM
    # llama_index.llms.base.LLM

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass


class Tool(ComponentView):
    # langchain ???
    # llama_index.tools.types.BaseTool

    @property
    @abstractmethod
    def tool_name(self) -> str:
        pass


class Agent(ComponentView):
    # langchain ???
    # llama_index.agent.types.BaseAgent

    @property
    @abstractmethod
    def agent_name(self) -> str:
        pass


class Memory(ComponentView):
    # langchain.schema.BaseMemory
    # llama_index ???
    pass


class Other(ComponentView):
    # Any component that does not fit into the other named categories.
    pass


class CustomComponent(ComponentView):
    class Custom(Other):
        # No categorization of custom class components for now. Using just one
        # "Custom" catch-all.

        @staticmethod
        def class_is(cls_obj: pyschema_utils.Class) -> bool:
            return True

    COMPONENT_VIEWS = [Custom]

    @staticmethod
    def constructor_of_class(
        cls_obj: pyschema_utils.Class,
    ) -> Type["CustomComponent"]:
        for view in CustomComponent.COMPONENT_VIEWS:
            if view.class_is(cls_obj):
                return view

        raise TypeError(f"Unknown custom component type with class {cls_obj}")

    @staticmethod
    def component_of_json(json: serial_utils.JSON) -> "CustomComponent":
        cls = pyschema_utils.Class.of_class_info(json)

        view = CustomComponent.constructor_of_class(cls)

        return view(json)

    @staticmethod
    def class_is(cls_obj: pyschema_utils.Class) -> bool:
        # Assumes this is the last check done.
        return True

    @classmethod
    def of_json(cls, json: serial_utils.JSON) -> "CustomComponent":
        return CustomComponent.component_of_json(json)


def instrumented_component_views(
    obj: object,
) -> Iterable[Tuple[serial_utils.Lens, ComponentView]]:
    """
    Iterate over contents of `obj` that are annotated with the CLASS_INFO
    attribute/key. Returns triples with the accessor/selector, the Class object
    instantiated from CLASS_INFO, and the annotated object itself.
    """

    for q, o in serial_utils.all_objects(obj):
        if (
            isinstance(o, pydantic.BaseModel)
            and constant_utils.CLASS_INFO in type(o).model_fields
        ):
            yield q, ComponentView.of_json(json=o)

        if isinstance(o, Dict) and constant_utils.CLASS_INFO in o:
            yield q, ComponentView.of_json(json=o)


def _can_import(to_import: str) -> bool:
    try:
        __import__(to_import)
        return True
    except ImportError:
        return False


class App(
    app_schema.AppDefinition,
    core_instruments.WithInstrumentCallbacks,
    Hashable,
):
    """Base app recorder type.

    Non-serialized fields here while the serialized ones are defined in
    [AppDefinition][trulens.core.schema.app.AppDefinition].

    This class is abstract. Use one of these concrete subclasses as appropriate:
    - [TruLlama][trulens.apps.llamaindex.TruLlama] for _LlamaIndex_ apps.
    - [TruChain][trulens.apps.langchain.TruChain] for _LangChain_ apps.
    - [TruRails][trulens.apps.nemo.TruRails] for _NeMo Guardrails_
        apps.
    - [TruVirtual][trulens.apps.virtual.TruVirtual] for recording
        information about invocations of apps without access to those apps.
    - [TruCustomApp][trulens.apps.custom.TruCustomApp] (To be deprecated in favor of TruApp) for custom
        apps. These need to be decorated to have appropriate data recorded.
    - [TruApp][trulens.apps.app.TruApp] for custom
        apps allowing maximized flexibility. These need to be decorated to have appropriate data recorded.
    - [TruBasicApp][trulens.apps.basic.TruBasicApp] for apps defined
        solely by a string-to-string method.
    """

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        # Tru, DB, most of the types on the excluded fields.
        arbitrary_types_allowed=True
    )

    feedbacks: List[core_feedback.Feedback] = pydantic.Field(
        exclude=True, default_factory=list
    )
    """Feedback functions to evaluate on each record."""

    session: core_session.TruSession = pydantic.Field(
        default_factory=core_session.TruSession, exclude=True
    )
    """Session for this app."""

    @property
    def connector(self) -> core_connector.DBConnector:
        """Database connector."""

        return self.session.connector

    @property
    def db(self) -> core_db.DB:
        """Database used by this app."""

        return self.connector.db

    @deprecation_utils.deprecated_property(
        "The `App.tru` property for retrieving `Tru` is deprecated. "
        "Use `App.connector` which contains the replacement `core_connector.DBConnector` class instead."
    )
    def tru(self) -> core_connector.DBConnector:
        return self.connector

    app: Any = pydantic.Field(exclude=True)
    """The app to be recorded."""

    main_method_name: Optional[str] = pydantic.Field(None)
    """Name of the main method of the app to be recorded. For serialization and this is required for OTEL."""

    instrument: Optional[core_instruments.Instrument] = pydantic.Field(
        None, exclude=True
    )
    """Instrumentation class.

    This is needed for serialization as it tells us which objects we want to be
    included in the json representation of this app.
    """

    recording_contexts: contextvars.ContextVar[
        core_instruments._RecordingContext
    ] = pydantic.Field(None, exclude=True)
    """Sequences of records produced by the this class used as a context manager
    are stored in a RecordingContext.

    Using a context var so that context managers can be nested.
    """

    instrumented_methods: Dict[int, Dict[Callable, serial_utils.Lens]] = (
        pydantic.Field(exclude=True, default_factory=dict)
    )
    """Mapping of instrumented methods (by id(.) of owner object and the
    function) to their path in this app."""

    records_with_pending_feedback_results: container_utils.BlockingSet[
        record_schema.Record
    ] = pydantic.Field(
        exclude=True, default_factory=container_utils.BlockingSet
    )
    """Records produced by this app which might have yet to finish
    feedback runs."""

    manage_pending_feedback_results_thread: Optional[threading_utils.Thread] = (
        pydantic.Field(exclude=True, default=None)
    )
    """Thread for manager of pending feedback results queue.

    See _manage_pending_feedback_results."""

    selector_check_warning: bool = False
    """Issue warnings when selectors are not found in the app with a placeholder
    record.

    If False, constructor will raise an error instead.
    """

    selector_nocheck: bool = False
    """Ignore selector checks entirely.

    This may be necessary 1if the expected record content cannot be determined
    before it is produced.
    """

    _context_vars_tokens: Dict[contextvars.ContextVar, contextvars.Token] = (
        pydantic.PrivateAttr(default_factory=dict)
    )

    snowflake_object_type: Optional[str] = pydantic.Field(
        None,
    )
    snowflake_object_name: Optional[str] = pydantic.Field(
        None,
    )
    snowflake_object_version: Optional[str] = pydantic.Field(
        None,
    )

    snowflake_run_dao: Optional[Any] = pydantic.Field(None, exclude=True)

    snowflake_app_dao: Optional[Any] = pydantic.Field(None, exclude=True)

    def __init__(
        self,
        connector: Optional[core_connector.DBConnector] = None,
        feedbacks: Optional[Iterable[core_feedback.Feedback]] = None,
        **kwargs,
    ):
        if feedbacks is not None:
            feedbacks = list(feedbacks)
        else:
            feedbacks = []

        # for us:
        if connector:
            tru_session = TruSession(connector=connector)
            if tru_session.connector is not connector:
                raise ValueError(
                    "Already created `TruSession` with different `connector`!"
                )
            kwargs["connector"] = connector

        kwargs["feedbacks"] = feedbacks
        kwargs["recording_contexts"] = contextvars.ContextVar(
            "recording_contexts", default=None
        )
        app = kwargs["app"]

        otel_enabled = TruSession().experimental_feature(
            core_experimental.Feature.OTEL_TRACING
        )
        main_method = None

        if otel_enabled:
            if app is None:
                raise ValueError(
                    "A valid app instance must be provided when specifying 'main_method'."
                )

            if "main_method" in kwargs:
                main_method = kwargs["main_method"]

                # Instead of always checking for binding,  enforce it except when app is an instance of TruWrapperApp (tru basic app).
                try:
                    from trulens.apps.basic import TruWrapperApp
                except ImportError:
                    TruWrapperApp = None

                if TruWrapperApp is None or not isinstance(app, TruWrapperApp):
                    if (
                        not hasattr(main_method, "__self__")
                        or main_method.__self__ != app
                    ):
                        raise ValueError(
                            f"main_method `{main_method.__name__}` must be bound to the provided `app` instance."
                        )

        super().__init__(**kwargs)

        if (
            is_otel_tracing_enabled()
            and self.feedback_mode
            != feedback_schema.FeedbackMode.WITH_APP_THREAD
        ):
            raise ValueError(
                "Cannot use `feedback_mode` other than `WITH_APP_THREAD` with OTel tracing!"
            )

        if main_method:
            self.main_method_name = main_method.__name__  # for serialization

        self._current_context_manager_lock = threading.Lock()
        self._current_context_manager = None

        self._evaluator = evaluator_utils.Evaluator(self)

        if connector and _can_import("trulens.connectors.snowflake"):
            from trulens.connectors.snowflake import SnowflakeConnector
            from trulens.connectors.snowflake.dao.enums import ObjectType

            if isinstance(connector, SnowflakeConnector):
                self.snowflake_object_type = (
                    ObjectType.EXTERNAL_AGENT.value
                    if "object_type" not in kwargs
                    or kwargs["object_type"] is None
                    else kwargs["object_type"]
                )

                (
                    self.snowflake_app_dao,
                    self.snowflake_run_dao,
                    self.snowflake_object_name,
                    self.snowflake_object_version,
                ) = connector.initialize_snowflake_dao_fields(
                    object_type=self.snowflake_object_type,
                    app_name=kwargs["app_name"],
                    app_version=kwargs["app_version"],
                )

        self.app = app

        if self.instrument is not None:
            self.instrument.instrument_object(
                obj=self.app, query=select_schema.Select.Query().app
            )
        else:
            pass

        if self.feedback_mode == feedback_schema.FeedbackMode.WITH_APP_THREAD:
            if is_otel_tracing_enabled():
                if kwargs.get("start_evaluator", True):
                    self.start_evaluator()
            else:
                self._start_manage_pending_feedback_results()

        if otel_enabled and main_method is not None:
            self._wrap_main_function(app, main_method.__name__)

        self._tru_post_init()

    def __del__(self):
        """Shut down anything associated with this app that might persist otherwise."""
        self.stop_evaluator()
        try:
            # Use object.__getattribute__ to avoid triggering __getattr__
            m_thread = object.__getattribute__(
                self, "manage_pending_feedback_results_thread"
            )
        except Exception:
            m_thread = None

        if m_thread is not None:
            try:
                records = object.__getattribute__(
                    self, "records_with_pending_feedback_results"
                )
                if records is not None:
                    records.shutdown()
            except Exception:
                # If records or shutdown is not available, ignore.
                pass

            try:
                m_thread.join()
            except Exception:
                pass

    @staticmethod
    def _has_instrumentation(func: Callable) -> bool:
        while hasattr(func, "__wrapped__"):
            if hasattr(func, TRULENS_INSTRUMENT_WRAPPER_FLAG) or hasattr(
                func, TRULENS_APP_SPECIFIC_INSTRUMENT_WRAPPER_FLAG
            ):
                return True
            func = func.__wrapped__
        return False

    @staticmethod
    def _has_record_root_instrumentation(func: Callable) -> bool:
        while hasattr(func, "__wrapped__"):
            if hasattr(func, TRULENS_RECORD_ROOT_INSTRUMENT_WRAPPER_FLAG):
                return True
            func = func.__wrapped__
        return False

    def _wrap_main_function(self, app: Any, method_name: str) -> None:
        if TruSession().experimental_feature(
            core_experimental.Feature.OTEL_TRACING, freeze=True
        ):
            from trulens.core.otel.instrument import instrument

            if not hasattr(app, method_name):
                raise ValueError(f"App must have an `{method_name}` method!")
            func = getattr(app, method_name)
            if self._has_instrumentation(func):
                return
            wrapper = instrument(is_app_specific_instrumentation=True)
            # HACK!: This is a major hack to get around the fact that we can't
            # set the desired method on the app object due to Pydantic only
            # allowing fields to be set on the class, not on the instance for
            # some reason. To get around this, we're setting it on the __dict__
            # of the app object, which is mutable and is the first place that
            # the field is looked up it seems. There's another implication of
            # this, which is that the desired method for this object will not
            # run whatever is instrumented by TruChain otherwise but that's
            # fine.
            app.__dict__[method_name] = wrapper(func)

    def _start_manage_pending_feedback_results(self) -> None:
        """Start the thread that manages the queue of records with
        pending feedback results.

        This is meant to be run permanently in a separate thread. It will
        remove records from the set `records_with_pending_feedback_results` as
        their feedback results are computed.
        """

        if self.manage_pending_feedback_results_thread is not None:
            raise RuntimeError("Manager Thread already started.")

        self.manage_pending_feedback_results_thread = threading.Thread(
            target=self._manage_pending_feedback_results,
            args=(weakref.proxy(self),),
            daemon=True,  # otherwise this thread will keep parent alive
            name=f"manage_pending_feedback_results_thread(app_name={self.app_name}, app_version={self.app_version})",
        )
        self.manage_pending_feedback_results_thread.start()

    @staticmethod
    def _manage_pending_feedback_results(
        self_proxy: weakref.ProxyType[App],
    ) -> None:
        """Manage the queue of records with pending feedback results.

        This is meant to be run permanently in a separate thread. It will
        remove records from the set records_with_pending_feedback_results as
        their feedback results are computed.
        """

        try:
            while True:
                record = self_proxy.records_with_pending_feedback_results.pop()
                record.wait_for_feedback_results()

        except StopIteration:
            pass
            # Set has been shut down.
        except ReferenceError:
            pass
            # self was unloaded, shut down as well.

    def wait_for_feedback_results(
        self, feedback_timeout: Optional[float] = None
    ) -> Iterable[record_schema.Record]:
        """Wait for all feedbacks functions to complete.

        Args:
            feedback_timeout: Timeout in seconds for waiting for feedback
                results for each feedback function. Note that this is not the
                total timeout for this entire blocking call.

        Returns:
            An iterable of records that have been waited on. Note a record will be
                included even if a feedback computation for it failed or
                timed out.

        This applies to all feedbacks on all records produced by this app. This
        call will block until finished and if new records are produced while
        this is running, it will include them.
        """
        if is_otel_tracing_enabled():
            raise RuntimeError(
                "`wait_for_feedback_results` is not supported with OTel tracing enabled, and is deprecated in favor of `retrieve_feedback_results`."
            )
        while (
            record := self.records_with_pending_feedback_results.pop(
                blocking=False
            )
        ) is not None:
            record.wait_for_feedback_results(feedback_timeout=feedback_timeout)
            yield record

    def retrieve_feedback_results(
        self, record_ids: Optional[List[str]] = None, timeout: float = 180
    ) -> pd.DataFrame:
        """Retrieve feedback results for all records in the app.

        Args:
            record_ids: List of record ids to retrieve feedback results for. If
                None, retrieves whatever results are available now.
            timeout: Timeout in seconds to wait.

        Returns:
            A dataframe with records as rows and feedbacks as columns.
        """
        TruSession().force_flush()
        if record_ids is not None:
            TruSession().wait_for_records(
                record_ids=record_ids, timeout=timeout
            )
            self._evaluator.compute_now(record_ids)
        records_df, feedback_cols = TruSession().get_records_and_feedback(
            record_ids=record_ids
        )
        records_df.set_index("record_id", inplace=True)
        return records_df[feedback_cols]

    @classmethod
    def select_context(cls, app: Optional[Any] = None) -> serial_utils.Lens:
        """Try to find retriever components in the given `app` and return a lens to
        access the retrieved contexts that would appear in a record were these
        components to execute."""

        # Catch the old case where a user calls App.select_context and gives
        # their app (not an App instance) as the app arg.
        if app is not None:
            mod = app.__class__.__module__
            if mod.startswith("langchain"):
                with import_utils.OptionalImports(
                    messages=optional_utils.REQUIREMENT_APPS_LANGCHAIN
                ):
                    from trulens.apps.langchain.tru_chain import TruChain

                return TruChain.select_context(app=app)
            elif mod.startswith("llama_index"):
                with import_utils.OptionalImports(
                    messages=optional_utils.REQUIREMENT_APPS_LLAMA
                ):
                    from trulens.apps.llamaindex.tru_llama import TruLlama

                return TruLlama.select_context(app=app)
            elif mod.startswith("nemoguardrails"):
                with import_utils.OptionalImports(
                    messages=optional_utils.REQUIREMENT_APPS_NEMO
                ):
                    from trulens.apps.nemo.tru_rails import TruRails

                return TruRails.select_context(app=app)
            else:
                raise ValueError(
                    f"Cannot determine the app type from its module {mod}."
                )

        raise NotImplementedError(
            f"`select_context` not implemented for {cls.__name__}. "
            "Call `select_context` using the appropriate subclass (TruChain, TruLlama, TruRails, etc)."
        )

    def __hash__(self):
        return hash(id(self))

    @staticmethod
    def _is_account_level_event_table_snowflake_connector(
        connector: Optional[core_connector.DBConnector],
    ) -> bool:
        if connector is None:
            return False
        try:
            from trulens.connectors.snowflake import SnowflakeConnector

            if not isinstance(connector, SnowflakeConnector):
                return False
            return connector.use_account_event_table
        except Exception:
            pass
        return False

    def _tru_post_init(self):
        """
        Database-related initialization and additional data checks.

        DB:
            - Insert the app into the database.
            - Insert feedback function definitions into the database.

        Checks:
            - In deferred mode, try to serialize and deserialize feedback functions.
            - Check that feedback function selectors are likely to refer to expected
                app or record components.

        """

        if self.connector is None:
            if self.feedback_mode != feedback_schema.FeedbackMode.NONE:
                logger.debug("Using default database connector.")
                self.connector = core_connector.DefaultDBConnector()

        else:
            if self.feedback_mode == feedback_schema.FeedbackMode.NONE:
                logger.warning(
                    "`connector` is specified but `feedback_mode` is `FeedbackMode.NONE`. "
                    "No feedback evaluation and logging will occur."
                )

        if self.connector is not None and not (
            self._is_account_level_event_table_snowflake_connector(
                self.connector
            )
            and is_otel_tracing_enabled()
        ):
            self.connector.add_app(app=self)

            if self.feedback_mode != feedback_schema.FeedbackMode.NONE:
                logger.debug("Inserting feedback function definitions to db.")

                for f in self.feedbacks:
                    self.connector.add_feedback_definition(f)

        else:
            if len(self.feedbacks) > 0:
                raise ValueError(
                    "Feedback logging requires `App.connector` to be specified."
                )

        for f in self.feedbacks:
            if (
                self.feedback_mode == feedback_schema.FeedbackMode.DEFERRED
                or f.run_location
                == feedback_schema.FeedbackRunLocation.SNOWFLAKE
            ):
                if (
                    isinstance(f.implementation, pyschema_utils.Method)
                    and f.implementation.obj.cls.module.module_name
                    == "trulens.providers.cortex.provider"
                    and f.implementation.obj.cls.name == "Cortex"
                ):
                    continue
                # Try to load each of the feedback implementations. Deferred
                # mode will do this but we want to fail earlier at app
                # constructor here.
                try:
                    f.implementation.load()
                except Exception as e:
                    raise Exception(
                        f"Feedback function {f} is not loadable. Cannot use DEFERRED feedback mode. {e}"
                    ) from e

        if is_otel_tracing_enabled():
            for feedback in self.feedbacks:
                feedback.check_otel_selectors()
        elif not self.selector_nocheck and not is_otel_tracing_enabled():
            dummy = self.dummy_record()

            for feedback in self.feedbacks:
                feedback.check_selectors(
                    app=self,
                    # Don't have a record yet, but use an empty one for the non-call related fields.
                    record=dummy,
                    warning=self.selector_check_warning,
                )

    def main_call(self, human: str) -> str:
        """If available, a single text to a single text invocation of this app."""

        if self.__class__.main_acall is not App.main_acall:
            # Use the async version if available.
            return asynchro_utils.sync(self.main_acall, human)

        raise NotImplementedError()

    async def main_acall(self, human: str) -> str:
        """If available, a single text to a single text invocation of this app."""

        if self.__class__.main_call is not App.main_call:
            logger.warning("Using synchronous version of main call.")
            # Use the sync version if available.
            return await asynchro_utils.desync(self.main_call, human)

        raise NotImplementedError()

    def _extract_content(self, value, content_keys=["content"]):
        """
        Extracts the 'content' from various data types commonly used by libraries
        like OpenAI, Canopy, LiteLLM, etc. This method navigates nested data
        structures (pydantic models, dictionaries, lists) to retrieve the
        'content' field. If 'content' is not directly available, it attempts to
        extract from known structures like 'choices' in a ChatResponse. This
        standardizes extracting relevant text or data from complex API responses
        or internal data representations.

        Args:
            value: The input data to extract content from. Can be a pydantic
                   model, dictionary, list, or basic data type.

        Returns:
            The extracted content, which may be a single value, a list of values,
            or a nested structure with content extracted from all levels.
        """
        return signature_utils._extract_content(
            value, content_keys=content_keys
        )

    def main_input(
        self, func: Callable, sig: Signature, bindings: BoundArguments
    ) -> str:
        """Determine (guess) the main input string for a main app call.

        Args:
            func: The main function we are targeting in this determination.

            sig: The signature of the above.

            bindings: The arguments to be passed to the function.

        Returns:
            The main input string.
        """
        return signature_utils.main_input(func, sig, bindings)

    def main_output(
        self,
        func: Callable,  # pylint: disable=W0613
        sig: Signature,  # pylint: disable=W0613
        bindings: BoundArguments,  # pylint: disable=W0613
        ret: Any,
    ) -> str:
        return signature_utils.main_output(func, ret)

    # WithInstrumentCallbacks requirement
    def on_method_instrumented(
        self, obj: object, func: Callable, path: serial_utils.Lens
    ):
        """Called by instrumentation system for every function requested to be
        instrumented by this app."""
        if is_otel_tracing_enabled():
            raise RuntimeError("Not supported with OTel tracing enabled!")
        if id(obj) in self.instrumented_methods:
            funcs = self.instrumented_methods[id(obj)]

            if func in funcs:
                old_path = funcs[func]

                if path != old_path:
                    logger.warning(
                        "Method %s was already instrumented on path %s. "
                        "Calls at %s may not be recorded.",
                        func,
                        old_path,
                        path,
                    )

                return

            else:
                funcs[func] = path

        else:
            funcs = dict()
            self.instrumented_methods[id(obj)] = funcs
            funcs[func] = path

    # WithInstrumentCallbacks requirement
    def get_methods_for_func(
        self, func: Callable
    ) -> Iterable[Tuple[int, Callable, serial_utils.Lens]]:
        """Get the methods (rather the inner functions) matching the given
        `func` and the path of each.

        See
        [WithInstrumentCallbacks.get_methods_for_func][trulens.core.instruments.WithInstrumentCallbacks.get_methods_for_func].
        """

        for _id, funcs in self.instrumented_methods.items():
            for f, path in funcs.items():
                if f == func:
                    yield (_id, f, path)

    # WithInstrumentCallbacks requirement
    def get_method_path(self, obj: object, func: Callable) -> serial_utils.Lens:
        """Get the path of the instrumented function `method` relative to this
        app."""

        # TODO: cleanup and/or figure out why references to objects change when executing langchain chains.

        funcs = self.instrumented_methods.get(id(obj))

        if funcs is None:
            logger.warning(
                "A new object of type %s at %s is calling an instrumented method %s. "
                "The path of this call may be incorrect.",
                python_utils.class_name(type(obj)),
                python_utils.id_str(obj),
                python_utils.callable_name(func),
            )
            try:
                _id, _, path = next(iter(self.get_methods_for_func(func)))

            except Exception:
                logger.warning(
                    "No other objects use this function so cannot guess path."
                )
                return None

            logger.warning(
                "Guessing path of new object is %s based on other object (%s) using this function.",
                path,
                python_utils.id_str(_id),
            )

            funcs = {func: path}

            self.instrumented_methods[id(obj)] = funcs

            return path

        else:
            if func not in funcs:
                logger.warning(
                    "A new object of type %s at %s is calling an instrumented method %s. "
                    "The path of this call may be incorrect.",
                    python_utils.class_name(type(obj)),
                    python_utils.id_str(obj),
                    python_utils.callable_name(func),
                )

                try:
                    _id, _, path = next(iter(self.get_methods_for_func(func)))
                except Exception:
                    logger.warning(
                        "No other objects use this function so cannot guess path."
                    )
                    return None

                logger.warning(
                    "Guessing path of new object is %s based on other object (%s) using this function.",
                    path,
                    python_utils.id_str(_id),
                )

                return path

            else:
                return funcs.get(func)

    def json(self, *args, **kwargs):
        """Create a json string representation of this app."""
        # Need custom jsonification here because it is likely the model
        # structure contains loops.

        return json_utils.json_str_of_obj(
            self, *args, instrument=self.instrument, **kwargs
        )

    def model_dump(self, *args, redact_keys: bool = False, **kwargs):
        # Same problem as in json.
        return json_utils.jsonify(
            self,
            instrument=self.instrument,
            redact_keys=redact_keys,
            *args,
            **kwargs,
        )

    def _prevent_invalid_otel_syntax(self):
        if self.session.experimental_feature(
            core_experimental.Feature.OTEL_TRACING
        ):
            raise RuntimeError("Invalid TruLens OTEL Tracing syntax.")

    # For use as a context manager.
    def __enter__(self):
        if self.session.experimental_feature(
            core_experimental.Feature.OTEL_TRACING
        ):
            from trulens.core.otel.instrument import OtelRecordingContext

            with self._current_context_manager_lock:
                if self._current_context_manager is not None:
                    raise RuntimeError(
                        "Already recording with a context manager, cannot nest!"
                    )
                self._current_context_manager = OtelRecordingContext(
                    tru_app=self,
                    app_name=self.app_name,
                    app_version=self.app_version,
                    run_name="",
                    input_id="",
                )
            return self._current_context_manager.__enter__()

        if not core_instruments.Instrument._have_context():
            raise RuntimeError(core_endpoint._NO_CONTEXT_WARNING)

        self._prevent_invalid_otel_syntax()
        ctx = core_instruments._RecordingContext(app=self)

        token = self.recording_contexts.set(ctx)
        ctx.token = token

        return ctx

    # For use as a context manager.
    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.session.experimental_feature(
            core_experimental.Feature.OTEL_TRACING
        ):
            with self._current_context_manager_lock:
                if self._current_context_manager is None:
                    raise RuntimeError("Unknown recording context manager!")
                context_manager = self._current_context_manager
                self._current_context_manager = None
            return context_manager.__exit__(exc_type, exc_value, exc_tb)

        self._prevent_invalid_otel_syntax()

        ctx = self.recording_contexts.get()
        self.recording_contexts.reset(ctx.token)

        if exc_type is not None:
            raise exc_value

        return

    # For use as a context manager.
    async def __aenter__(self):
        self._prevent_invalid_otel_syntax()

        ctx = core_instruments._RecordingContext(app=self)

        token = self.recording_contexts.set(ctx)
        ctx.token = token

        # self._set_context_vars()

        return ctx

    # For use as a context manager.
    async def __aexit__(self, exc_type, exc_value, exc_tb):
        self._prevent_invalid_otel_syntax()

        ctx = self.recording_contexts.get()
        self.recording_contexts.reset(ctx.token)

        # self._reset_context_vars()

        if exc_type is not None:
            raise exc_value

        return

    def _set_context_vars(self):
        # HACK: For debugging purposes, try setting/resetting all context vars
        # used in trulens around the app context managers due to bugs in trying
        # to set/reset them where more appropriate. This is not ideal as not
        # resetting context vars where appropriate will result possibly in
        # incorrect tracing information.

        from trulens.core.feedback.endpoint import Endpoint
        from trulens.core.instruments import WithInstrumentCallbacks

        CONTEXT_VARS = [
            WithInstrumentCallbacks._stack_contexts,
            WithInstrumentCallbacks._context_contexts,
            Endpoint._context_endpoints,
        ]

        for var in CONTEXT_VARS:
            self._context_vars_tokens[var] = var.set(var.get())

    def _reset_context_vars(self):
        # HACK: See _set_context_vars.
        for var, token in self._context_vars_tokens.items():
            var.reset(token)

        del self._context_vars_tokens[var]

    # WithInstrumentCallbacks requirement
    def on_new_record(
        self, func
    ) -> Iterable[core_instruments._RecordingContext]:
        """Called at the start of record creation.

        See
        [WithInstrumentCallbacks.on_new_record][trulens.core.instruments.WithInstrumentCallbacks.on_new_record].
        """
        if is_otel_tracing_enabled():
            raise RuntimeError("Not supported with OTel tracing enabled!")
        ctx = self.recording_contexts.get(contextvars.Token.MISSING)

        while ctx is not contextvars.Token.MISSING:
            yield ctx
            ctx = ctx.token.old_value

    # WithInstrumentCallbacks requirement
    def on_add_record(
        self,
        ctx: core_instruments._RecordingContext,
        func: Callable,
        sig: Signature,
        bindings: BoundArguments,
        ret: Any,
        error: Any,
        perf: base_schema.Perf,
        cost: base_schema.Cost,
        existing_record: Optional[record_schema.Record] = None,
        final: bool = False,
    ) -> record_schema.Record:
        """Called by instrumented methods if they use _new_record to construct a
        "record call list.

        See
        [WithInstrumentCallbacks.on_add_record][trulens.core.instruments.WithInstrumentCallbacks.on_add_record].
        """
        if is_otel_tracing_enabled():
            raise RuntimeError("Not supported with OTel tracing enabled!")

        def build_record(
            calls: Iterable[record_schema.RecordAppCall],
            record_metadata: serial_utils.JSON,
            existing_record: Optional[record_schema.Record] = None,
        ) -> record_schema.Record:
            calls = list(calls)

            assert len(calls) > 0, "No information recorded in call."

            if bindings is not None:
                if existing_record is None:
                    main_in = json_utils.jsonify(
                        self.main_input(func, sig, bindings)
                    )
                else:
                    main_in = existing_record.main_input
            else:
                main_in = None

            if error is None:
                assert bindings is not None, "No bindings despite no error."
                if final:
                    main_out = self.main_output(func, sig, bindings, ret)
                else:
                    main_out = f"TruLens: Record not yet finalized: {ret}"
            else:
                main_out = None

            updates = dict(
                main_input=main_in,
                main_output=json_utils.jsonify(main_out),
                main_error=json_utils.jsonify(error),
                calls=calls,
                cost=cost,
                perf=perf,
                app_id=self.app_id,
                tags=self.tags,
                meta=json_utils.jsonify(record_metadata),
            )

            if existing_record is not None:
                existing_record.update(**updates)
            else:
                existing_record = record_schema.Record(**updates)

            return existing_record

        # Finishing record needs to be done in a thread lock, done there:
        record = ctx.finish_record(
            build_record, existing_record=existing_record
        )

        if error is not None:
            # May block on DB.
            self._handle_error(record=record, error=error)
            raise error

        # Only continue with the feedback steps if the record is final.
        if not final:
            return record

        # Will block on DB, but not on feedback evaluation, depending on
        # FeedbackMode:
        record.feedback_and_future_results = self._handle_record(record=record)
        if record.feedback_and_future_results is not None:
            record.feedback_results = [
                tup[1] for tup in record.feedback_and_future_results
            ]

        if record.feedback_and_future_results is None:
            return record

        if self.feedback_mode == feedback_schema.FeedbackMode.WITH_APP_THREAD:
            # Add the record to ones with pending feedback.

            self.records_with_pending_feedback_results.add(record)

        elif self.feedback_mode == feedback_schema.FeedbackMode.WITH_APP:
            # If in blocking mode ("WITH_APP"), wait for feedbacks to finished
            # evaluating before returning the record.

            record.wait_for_feedback_results()

        return record

    def _check_instrumented(self, func):
        """
        Issue a warning and some instructions if a function that has not been
        instrumented is being used in a `with_` call.
        """
        if is_otel_tracing_enabled():
            raise RuntimeError("Not supported with OTel tracing enabled!")

        if not isinstance(func, Callable):
            raise TypeError(
                f"Expected `func` to be a callable, but got {python_utils.class_name(type(func))}."
            )

        # If func is actually an object that implements __call__, check __call__
        # instead.
        if not (inspect.isfunction(func) or inspect.ismethod(func)):
            func = func.__call__

        if not python_utils.safe_hasattr(
            func, core_instruments.Instrument.INSTRUMENT
        ):
            if core_instruments.Instrument.INSTRUMENT in dir(func):
                # HACK009: Need to figure out the __call__ accesses by class
                # name/object name with relation to this check for
                # instrumentation because we keep hitting spurious warnings
                # here. This is a temporary workaround.
                return

            logger.warning(
                """
Function %s has not been instrumented. This may be ok if it will call a function
that has been instrumented exactly once. Otherwise unexpected results may
follow. You can use `AddInstruments.method` of `trulens.core.instruments` before
you use the `%s` wrapper to make sure `%s` does get instrumented. `%s` method
`print_instrumented` may be used to see methods that have been instrumented.
""",
                func,
                python_utils.class_name(self),
                python_utils.callable_name(func),
                python_utils.class_name(self),
            )

    async def awith_(
        self, func: asynchro_utils.CallableMaybeAwaitable[A, T], *args, **kwargs
    ) -> T:
        """Call the given async `func` with the given `*args` and `**kwargs`
        while recording, producing `func` results.

        The record of the computation is available through other means like the
        database or dashboard. If you need a record of this execution
        immediately, you can use `awith_record` or the `App` as a context
        manager instead.
        """

        res, _ = await self.awith_record(func, *args, **kwargs)

        return res

    async def with_(self, func: Callable[[A], T], *args, **kwargs) -> T:
        """Call the given async `func` with the given `*args` and `**kwargs`
        while recording, producing `func` results.

        The record of the computation is available through other means like the
        database or dashboard. If you need a record of this execution
        immediately, you can use `awith_record` or the `App` as a context
        manager instead.
        """

        res, _ = self.with_record(func, *args, **kwargs)

        return res

    def with_record(
        self,
        func: Callable[[A], T],
        *args,
        record_metadata: serial_utils.JSON = None,
        **kwargs,
    ) -> Tuple[T, record_schema.Record]:
        """
        Call the given `func` with the given `*args` and `**kwargs`, producing
        its results as well as a record of the execution.
        """
        if is_otel_tracing_enabled():
            raise RuntimeError("Not supported with OTel tracing enabled!")

        if not isinstance(func, Callable):
            if hasattr(func, "__call__"):
                func = func.__call__

        self._check_instrumented(func)

        with self as ctx:
            ctx.record_metadata = record_metadata
            ret = func(*args, **kwargs)

        assert len(ctx.records) > 0, (
            f"Did not create any records. "
            f"This means that no instrumented methods were invoked in the process of calling {func}."
        )

        return ret, ctx.get()

    async def awith_record(
        self,
        func: Callable[[A], Awaitable[T]],
        *args,
        record_metadata: serial_utils.JSON = None,
        **kwargs,
    ) -> Tuple[T, record_schema.Record]:
        """
        Call the given `func` with the given `*args` and `**kwargs`, producing
        its results as well as a record of the execution.
        """
        if is_otel_tracing_enabled():
            raise RuntimeError("Not supported with OTel tracing enabled!")

        if not isinstance(func, Callable):
            if hasattr(func, "__call__"):
                func = func.__call__

        self._check_instrumented(func)

        async with self as ctx:
            ctx.record_metadata = record_metadata
            ret = await func(*args, **kwargs)

        assert len(ctx.records) > 0, (
            f"Did not create any records. "
            f"This means that no instrumented methods were invoked in the process of calling {func}."
        )

        return ret, ctx.get()

    def _throw_dep_message(
        self, method, is_async: bool = False, with_record: bool = False
    ):
        # Raises a deprecation message for the various methods that pass through to
        # wrapped app while recording.

        cname = self.__class__.__name__

        iscall = method == "__call__"

        old_method = f"""{method}{"_with_record" if with_record else ""}"""
        if iscall:
            old_method = f"""call{"_with_record" if with_record else ""}"""
        new_method = f"""{"a" if is_async else ""}with_{"record" if with_record else ""}"""

        app_callable = f"""app.{method}"""
        if iscall:
            app_callable = "app"

        raise AttributeError(
            f"""
`{old_method}` is deprecated; To record results of your app's execution, use one of these options to invoke your app:
    (1) Use the `{"a" if is_async else ""}with_{"record" if with_record else ""}` method:
        ```python
        app # your app
        tru_app_recorder: {cname} = {cname}(app, ...)
        result{", record" if with_record else ""} = {"await " if is_async else ""}tru_app_recorder.{new_method}({app_callable}, ...args/kwargs-to-{app_callable}...)
        ```
    (2) Use {cname} as a context manager:
        ```python
        app # your app
        tru_app_recorder: {cname} = {cname}(app, ...)
        with tru_app_recorder{" as records" if with_record else ""}:
            result = {"await " if is_async else ""}{app_callable}(...args/kwargs-to-{app_callable}...)
        {"record = records.get()" if with_record else ""}
        ```
"""
        )

    def _add_future_feedback(
        self,
        future_or_result: Union[
            feedback_schema.FeedbackResult,
            Future[feedback_schema.FeedbackResult],
        ],
    ) -> None:
        """
        Callback used to add feedback results to the database once they are
        done.

        See [_handle_record][trulens.core.app.App._handle_record].
        """
        if is_otel_tracing_enabled():
            raise RuntimeError("Not supported with OTel tracing enabled!")

        if isinstance(future_or_result, Future):
            res = future_or_result.result()
        else:
            res = future_or_result

        self.connector.add_feedback(res)

    def _handle_record(
        self,
        record: record_schema.Record,
        feedback_mode: Optional[feedback_schema.FeedbackMode] = None,
    ) -> Optional[
        List[
            Tuple[
                core_feedback.Feedback,
                Future[feedback_schema.FeedbackResult],
            ]
        ]
    ]:
        """
        Write out record-related info to database if set and schedule feedback
        functions to be evaluated. If feedback_mode is provided, will use that
        mode instead of the one provided to constructor.
        """

        if feedback_mode is None:
            feedback_mode = self.feedback_mode

        if self.feedback_mode is None:
            return None

        # If in buffered mode, call add record nowait.
        if self.record_ingest_mode == app_schema.RecordIngestMode.BUFFERED:
            self.connector.add_record_nowait(record=record)
            return

        # Need to add record to db before evaluating feedback functions.
        record_id = self.connector.add_record(record=record)

        if len(self.feedbacks) == 0:
            return []

        if feedback_mode == feedback_schema.FeedbackMode.NONE:
            # Do not run any feedbacks in this case (now or deferred).
            return None

        if feedback_mode == feedback_schema.FeedbackMode.DEFERRED:
            # Run all feedbacks as deferred.
            deferred_feedbacks = self.feedbacks
            undeferred_feedbacks = []
        else:
            # Run only the feedbacks to be run in Snowflake as deferred.
            deferred_feedbacks = []
            undeferred_feedbacks = []
            for f in self.feedbacks:
                if (
                    f.run_location
                    == feedback_schema.FeedbackRunLocation.SNOWFLAKE
                ):
                    deferred_feedbacks.append(f)
                else:
                    undeferred_feedbacks.append(f)

        # Insert into the feedback table the deferred feedbacks.
        for f in deferred_feedbacks:
            self.connector.db.insert_feedback(
                feedback_schema.FeedbackResult(
                    name=f.name,
                    record_id=record_id,
                    feedback_definition_id=f.feedback_definition_id,
                )
            )
        # Compute the undeferred feedbacks.
        return self._submit_feedback_functions(
            record=record,
            feedback_functions=undeferred_feedbacks,
            app=self,
            connector=self.connector,
            on_done=self._add_future_feedback,
        )

    def _handle_error(self, record: record_schema.Record, error: Exception):
        if self.connector is None:
            return

    def __getattr__(self, __name: str) -> Any:
        # A message for cases where a user calls something that the wrapped app
        # contains. We do not support this form of pass-through calls anymore.

        try:
            # Some odd interaction with pydantic.PrivateAttr causes this handler
            # to be called for private attributes even though they exist. So we
            # double check here with pydantic's getattr.
            return pydantic.BaseModel.__getattr__(self, __name)
        except AttributeError:
            pass

        if __name == "app":
            return None

        if python_utils.safe_hasattr(self.app, __name):
            msg = ATTRIBUTE_ERROR_MESSAGE.format(
                attribute_name=__name,
                class_name=type(self).__name__,
                app_class_name=type(self.app).__name__,
            )
            raise AttributeError(msg)

        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{__name}'"
            )

    def dummy_record(
        self,
        cost: base_schema.Cost = base_schema.Cost(),
        perf: base_schema.Perf = base_schema.Perf.now(),
        ts: datetime.datetime = datetime.datetime.now(),
        main_input: str = "main_input are strings.",
        main_output: str = "main_output are strings.",
        main_error: str = "main_error are strings.",
        meta: Dict = {"metakey": "meta are dicts"},
        tags: str = "tags are strings",
    ) -> record_schema.Record:
        """Create a dummy record with some of the expected structure without
        actually invoking the app.

        The record is a guess of what an actual record might look like but will
        be missing information that can only be determined after a call is made.

        All args are [Record][trulens.core.schema.record.Record] fields except these:

            - `record_id` is generated using the default id naming schema.
            - `app_id` is taken from this recorder.
            - `calls` field is constructed based on instrumented methods.
        """
        if is_otel_tracing_enabled():
            raise RuntimeError("Not supported with OTel tracing enabled!")

        calls = []

        for methods in self.instrumented_methods.values():
            for func, lens in methods.items():
                component = lens.get_sole_item(self)

                if not hasattr(component, func.__name__):
                    continue
                method = getattr(component, func.__name__)

                sig = inspect.signature(method)

                method_serial = pyschema_utils.FunctionOrMethod.of_callable(
                    method
                )

                sample_args = {}
                for p in sig.parameters.values():
                    if p.default == inspect.Parameter.empty:
                        sample_args[p.name] = None
                    else:
                        sample_args[p.name] = p.default

                sample_call = record_schema.RecordAppCall(
                    stack=[
                        record_schema.RecordAppCallMethod(
                            path=lens, method=method_serial
                        )
                    ],
                    args=sample_args,
                    rets=None,
                    pid=0,
                    tid=0,
                )

                calls.append(sample_call)

        return record_schema.Record(
            app_id=self.app_id,
            calls=calls,
            cost=cost,
            perf=perf,
            ts=ts,
            main_input=main_input,
            main_output=main_output,
            main_error=main_error,
            meta=meta,
            tags=tags,
        )

    def instrumented(self) -> Iterable[Tuple[serial_utils.Lens, ComponentView]]:
        """
        Iteration over instrumented components and their categories.
        """
        if is_otel_tracing_enabled():
            raise RuntimeError("Not supported with OTel tracing enabled!")

        for q, c in instrumented_component_views(self.model_dump()):
            # Add the chain indicator so the resulting paths can be specified
            # for feedback selectors.
            q = serial_utils.Lens(
                path=(
                    serial_utils.GetItemOrAttribute(
                        item_or_attribute="__app__"
                    ),
                )
                + q.path
            )
            yield q, c

    def print_instrumented(self) -> None:
        """Print the instrumented components and methods."""

        print("Components:")
        self.print_instrumented_components()
        print("\nMethods:")
        self.print_instrumented_methods()

    def format_instrumented_methods(self) -> str:
        """Build a string containing a listing of instrumented methods."""

        return "\n".join(
            f"Object at 0x{obj:x}:\n\t"
            + "\n\t".join(
                f"{m} with path {select_schema.Select.App + path}"
                for m, path in p.items()
            )
            for obj, p in self.instrumented_methods.items()
        )

    def print_instrumented_methods(self) -> None:
        """Print instrumented methods."""

        logger.info(self.format_instrumented_methods())

    def print_instrumented_components(self) -> None:
        """Print instrumented components and their categories."""

        object_strings = []

        for t in self.instrumented():
            path = serial_utils.Lens(t[0].path[1:])
            obj = next(iter(path.get(self)))
            object_strings.append(
                f"\t{type(obj).__name__} ({t[1].__class__.__name__}) at 0x{id(obj):x} with path {str(t[0])}"
            )

        logger.info("\n".join(object_strings))

    def _check_snowflake_dao(self):
        if (
            not hasattr(self, "snowflake_app_dao")
            or self.snowflake_app_dao is None
        ):
            msg = (
                "This API requires a Snowpark session to initialize snowflake-specific DAO instance. Please initialize App with "
                "object_type='EXTERNAL AGENT' and a valid snowpark_session."
            )
            logger.error(msg)
            raise ValueError(msg)

    def add_run(self, run_config: RunConfig) -> Union[Run, None]:
        """add a new run to the snowflake App (if not already exists)

        Args:
            run_config (RunConfig):  Run config
            input_df (Optional[pd.DataFrame]): optional input dataset

        Returns:
            Run: Run instance
        """

        self._check_snowflake_dao()

        try:
            run_metadata_df = self.snowflake_run_dao.create_new_run(
                object_name=self.snowflake_object_name,
                object_type=self.snowflake_object_type,
                object_version=self.snowflake_object_version,
                run_name=run_config.run_name,
                dataset_name=run_config.dataset_name,
                source_type=run_config.source_type,
                dataset_spec=validate_dataset_spec(run_config.dataset_spec),
                description=run_config.description,
                label=run_config.label,
                llm_judge_name=run_config.llm_judge_name,
            )

            return Run.from_metadata_df(
                run_metadata_df,
                {
                    "app": self,
                    "main_method_name": self.main_method_name,
                    "run_dao": self.snowflake_run_dao,
                    "tru_session": self.session,
                },
            )

        except Exception as e:
            if "already exists." in str(e):
                logger.debug(f"Run {run_config.run_name} already exists.")
                return self.get_run(run_config.run_name)
            else:
                logger.error(f"Failed to add run {run_config.run_name}. {e}")
                raise

    def get_run(self, run_name: str) -> Run:
        """Retrieve a run by name.

        Args:
            run_name (str): unique name of the run

        Returns:
            Run: Run instance
        """
        self._check_snowflake_dao()
        run_metadata_df = self.snowflake_run_dao.get_run(
            run_name=run_name,
            object_name=self.snowflake_object_name,
            object_type=self.snowflake_object_type,
            object_version=self.snowflake_object_version,
        )
        if run_metadata_df.empty:
            raise ValueError(f"Run {run_name} not found.")

        return Run.from_metadata_df(
            run_metadata_df,
            {
                "app": self,
                "main_method_name": self.main_method_name,
                "run_dao": self.snowflake_run_dao,
                "tru_session": self.session,
            },
        )

    def list_runs(self) -> List[Run]:
        """Retrieve all runs belong to the snowflake App.

        Returns:
            List[Run]: List of Run instances
        """
        self._check_snowflake_dao()
        run_metadata_df = self.snowflake_run_dao.list_all_runs(
            object_name=self.snowflake_object_name,
            object_type=self.snowflake_object_type,
        )
        runs = []

        if run_metadata_df.empty:
            return runs

        all_runs_lst = json.loads(run_metadata_df.iloc[0].iloc[-1])

        # return all_runs_lst
        for run_dict in all_runs_lst:
            runs.append(
                Run.from_metadata_df(
                    pd.DataFrame({"col": [json.dumps(run_dict)]}),
                    {
                        "app": self.app,
                        "main_method_name": self.main_method_name,
                        "run_dao": self.snowflake_run_dao,
                        "tru_session": self.session,
                        "object_name": self.snowflake_object_name,
                        "object_type": self.snowflake_object_type,
                    },
                )
            )
        return runs

    def delete(self) -> None:
        """Delete the snowflake App (external agent) in snowflake. All versions will be removed"""
        self._check_snowflake_dao()
        self.snowflake_app_dao.drop_agent(self.snowflake_object_name)

    def delete_version(self) -> None:
        """Delete the current version of the snowflake App (external agent) in snowflake.
        Only the non-default version can be deleted.
        """
        self._check_snowflake_dao()
        self.snowflake_app_dao.drop_current_version(self.snowflake_object_name)

    def run(self, run_name: str):
        if self.session.experimental_feature(
            core_experimental.Feature.OTEL_TRACING
        ):
            from trulens.core.otel.instrument import OtelRecordingContext

            return OtelRecordingContext(
                tru_app=self,
                app_name=self.app_name,
                app_version=self.app_version,
                run_name=run_name,
                input_id=None,
            )
        raise NotImplementedError(
            "This feature is not yet implemented for non-OTEL TruLens!"
        )

    def input(self, input_id: str):
        if self.session.experimental_feature(
            core_experimental.Feature.OTEL_TRACING
        ):
            from trulens.core.otel.instrument import OtelRecordingContext

            return OtelRecordingContext(
                tru_app=self,
                app_name=self.app_name,
                app_version=self.app_version,
                run_name=None,
                input_id=input_id,
            )
        raise NotImplementedError(
            "This feature is not yet implemented for non-OTEL TruLens!"
        )

    def instrumented_invoke_main_method(
        self,
        run_name: str,
        input_id: str,
        input_records_count: Optional[
            int
        ] = None,  # total expected number of input records for the run
        ground_truth_output: Optional[str] = None,
        main_method_args: Optional[Sequence[Any]] = None,
        main_method_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if self.session.experimental_feature(
            core_experimental.Feature.OTEL_TRACING
        ):
            from trulens.core.otel.instrument import OtelRecordingContext

            with OtelRecordingContext(
                tru_app=self,
                app_name=self.app_name,
                app_version=self.app_version,
                run_name=run_name,
                input_id=input_id,
                input_records_count=input_records_count,
                ground_truth_output=ground_truth_output,
            ):
                f = getattr(self.app, self.main_method_name)
                if main_method_args is None:
                    main_method_args = ()
                if main_method_kwargs is None:
                    main_method_kwargs = {}
                return f(*main_method_args, **main_method_kwargs)
        raise NotImplementedError(
            "This feature is not yet implemented for non-OTEL TruLens!"
        )

    def compute_feedbacks(
        self,
        raise_error_on_no_feedbacks_computed: bool = True,
        events: Optional[pd.DataFrame] = None,
    ) -> None:
        """Compute feedbacks for the app.

        Args:
            raise_error_on_no_feedbacks_computed:
                Raise an error if no feedbacks were computed. Default is True.
            events:
                The events to compute feedbacks from. If None, uses all
                events from the app.
        """
        if not is_otel_tracing_enabled():
            raise ValueError(
                "This method is only supported for OTEL Tracing. Please enable OTEL tracing in the environment!"
            )
        if events is None:
            # Get all events associated with this app name and version.
            # TODO(otel): Should probably handle the case where there are a lot of events with pagination.
            events = self.connector.get_events(
                app_name=self.app_name, app_version=self.app_version
            )
        for feedback in self.feedbacks:
            compute_feedback_by_span_group(
                events,
                feedback.name,
                feedback.imp,
                feedback.higher_is_better,
                feedback.selectors,
                feedback.aggregator,
                raise_error_on_no_feedbacks_computed,
            )

    def start_evaluator(self) -> None:
        """Start the evaluator for the app."""
        self._evaluator.start_evaluator()

    def stop_evaluator(self) -> None:
        """Stop the evaluator for the app."""
        if hasattr(self, "_evaluator") and self._evaluator is not None:
            self._evaluator.stop_evaluator()


# NOTE: Cannot App.model_rebuild here due to circular imports involving mod_session.TruSession
# and database.base.DB. Will rebuild each App subclass instead.
