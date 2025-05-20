"""Serializable app-related classes."""

from __future__ import annotations

from enum import Enum
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import warnings

import dill
import pydantic
from trulens.core.feedback import feedback as core_feedback
import trulens.core.instruments as core_instruments
from trulens.core.schema import base as base_schema
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import record as record_schema
from trulens.core.schema import select as select_schema
from trulens.core.schema import types as types_schema
from trulens.core.utils import json as json_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import text as text_utils
from trulens.core.utils import threading as threading_utils

if TYPE_CHECKING:
    from trulens.core._utils.pycompat import Future  # import style exception
    from trulens.core.database import connector as db_connector
    from trulens.core.feedback import feedback as core_feedback

logger = logging.getLogger(__name__)


class RecordIngestMode(str, Enum):
    """Mode of records ingestion.

    Specify this using the `ingest_mode` to [App][trulens.core.app.App] constructors.
    """

    IMMEDIATE = "immediate"
    """Each record is ingested one by one and written to the database. This is the default mode."""

    BUFFERED = "buffered"
    """Records are buffered and ingested in batches to the database."""


class AppDefinition(pyschema_utils.WithClassInfo, serial_utils.SerialModel):
    """Serialized fields of an app here whereas [App][trulens.core.app.App]
    contains non-serialized fields."""

    app_id: types_schema.AppID = pydantic.Field(frozen=True)  # str
    """Unique identifier for this app.

    Computed deterministically from app_name and app_version. Leaving it here
    for it to be dumped when serializing. Also making it read-only as it should
    not be changed after creation.
    """

    app_name: types_schema.AppName  # str
    """Name for this app. Default is "default_app"."""

    app_version: types_schema.AppVersion  # str
    """Version tag for this app. Default is "base"."""

    tags: types_schema.Tags  # str
    """Tags for the app."""

    metadata: (
        types_schema.Metadata
    )  # dict  # TODO: rename to meta for consistency with other metas
    """Metadata for the app."""

    feedback_definitions: Sequence[types_schema.FeedbackDefinitionID] = []
    """Feedback functions to evaluate on each record."""

    feedback_mode: feedback_schema.FeedbackMode = (
        feedback_schema.FeedbackMode.WITH_APP_THREAD
    )
    """How to evaluate feedback functions upon producing a record."""

    record_ingest_mode: RecordIngestMode
    """Mode of records ingestion."""

    root_class: pyschema_utils.Class
    """Class of the main instrumented object.

    Ideally this would be a [ClassVar][typing.ClassVar] but since we want to check this without
    instantiating the subclass of
    [AppDefinition][trulens.core.schema.app.AppDefinition] that would define it, we
    cannot use [ClassVar][typing.ClassVar].
    """

    root_callable: ClassVar[pyschema_utils.FunctionOrMethod]
    """App's main method.

    This is to be filled in by subclass.
    """

    app: serial_utils.JSONized[AppDefinition]
    """Wrapped app in jsonized form."""

    initial_app_loader_dump: Optional[serial_utils.SerialBytes] = None
    """Serialization of a function that loads an app.

    Dump is of the initial app state before any invocations. This can be used to
    create a new session.

    Warning:
        Experimental work in progress.
    """

    app_extra_json: serial_utils.JSON
    """Info to store about the app and to display in dashboard.

    This can be used even if app itself cannot be serialized. `app_extra_json`,
    then, can stand in place for whatever data the user might want to keep track
    of about the app.
    """

    def __init__(
        self,
        app_id: Optional[types_schema.AppName] = None,
        app_name: Optional[types_schema.AppName] = None,
        app_version: Optional[types_schema.AppVersion] = None,
        tags: Optional[types_schema.Tags] = None,
        metadata: Optional[types_schema.Metadata] = None,
        feedback_mode: feedback_schema.FeedbackMode = feedback_schema.FeedbackMode.WITH_APP_THREAD,
        record_ingest_mode: RecordIngestMode = RecordIngestMode.IMMEDIATE,
        app_extra_json: serial_utils.JSON = None,
        **kwargs,
    ):
        kwargs["app_name"] = str(app_name or app_id or "default_app")
        kwargs["app_version"] = app_version or "base"
        kwargs["feedback_mode"] = feedback_mode
        kwargs["tags"] = ""
        kwargs["metadata"] = metadata or {}
        kwargs["app_extra_json"] = app_extra_json or dict()
        kwargs["feedback_definitions"] = [
            f.feedback_definition_id for f in kwargs.get("feedbacks", [])
        ]
        kwargs["record_ingest_mode"] = record_ingest_mode
        kwargs["app_id"] = self._compute_app_id(
            kwargs["app_name"], kwargs["app_version"]
        )
        if app_id is not None and kwargs["app_id"] != app_id:
            # Warning for now:
            warnings.warn(
                "Custom `app_id` values are deprecated. Use `app_name` and/or `app_version` instead.",
                DeprecationWarning,
            )
            # After dep period, change to this:
            # raise ValueError(
            #    "`app_id` does not match with `app_name` and `app_version`! `app_id` is auto-generated and should not need to be passed in!"
            # )

        super().__init__(**kwargs)

        self.record_ingest_mode = record_ingest_mode

        if tags is None:
            tags = "-"  # Set tags to a "-" if None is provided
        self.tags = tags

        # EXPERIMENTAL
        if "initial_app_loader" in kwargs:
            try:
                dump = dill.dumps(kwargs["initial_app_loader"], recurse=True)

                if len(dump) > base_schema.MAX_DILL_SIZE:
                    logger.warning(
                        f"`initial_app_loader` dump is too big ({text_utils.format_quantity(len(dump))}) > {text_utils.format_quantity(base_schema.MAX_DILL_SIZE)} bytes). "
                        "If you are loading large objects, include the loading logic inside `initial_app_loader`.",
                    )
                else:
                    self.initial_app_loader_dump = serial_utils.SerialBytes(
                        data=dump
                    )

                    # This is an older serialization approach that saved things
                    # in local files instead of the DB. Leaving here for now as
                    # serialization of large apps might make this necessary
                    # again.
                    # path_json = Path.cwd() / f"{app_id}.json"
                    # path_dill = Path.cwd() / f"{app_id}.dill"

                    # with path_json.open("w") as fh:
                    #     fh.write(json_utils.json_str_of_obj(self))

                    # with path_dill.open("wb") as fh:
                    #     fh.write(dump)

                    # print(f"Wrote loadable app to {path_json} and {path_dill}.")

            except Exception as e:
                logger.warning(
                    "Could not serialize app loader. "
                    "Some trulens features may not be available: %s",
                    e,
                )

    @staticmethod
    def continue_session(
        app_definition_json: serial_utils.JSON, app: Any
    ) -> AppDefinition:
        # initial_app_loader: Optional[Callable] = None) -> 'AppDefinition':
        """Instantiate the given `app` with the given state
        `app_definition_json`.

        Warning:
            This is an experimental feature with ongoing work.

        Args:
            app_definition_json: The json serialized app.

            app: The app to continue the session with.

        Returns:
            A new `AppDefinition` instance with the given `app` and the given
                `app_definition_json` state.
        """

        app_definition_json["app"] = app

        cls = pyschema_utils.WithClassInfo.get_class(app_definition_json)

        return cls(**app_definition_json)

    @staticmethod
    def _compute_app_id(app_name: str, app_version: str) -> str:
        # Generate a deterministic hash-based ID based on app_name and
        # app_version.
        # NOTE: there is an ongoing discussion about whether to use app_id or
        # directly use/index on the unique combination of
        # (app_name, app_version) when querying the database.
        return json_utils.obj_id_of_obj(
            obj={"app_name": app_name, "app_version": app_version}, prefix="app"
        )

    @staticmethod
    def new_session(
        app_definition_json: serial_utils.JSON,
        initial_app_loader: Optional[Callable] = None,
    ) -> AppDefinition:
        """Create an app instance at the start of a session.

        Warning:
            This is an experimental feature with ongoing work.

        Create a copy of the json serialized app with the enclosed app being
        initialized to its initial state before any records are produced (i.e.
        blank memory).
        """

        serial_bytes_json: Optional[serial_utils.JSON] = app_definition_json[
            "initial_app_loader_dump"
        ]

        if initial_app_loader is None:
            assert (
                serial_bytes_json is not None
            ), "Cannot create new session without `initial_app_loader`."

            serial_bytes = serial_utils.SerialBytes.model_validate(
                serial_bytes_json
            )

            app = dill.loads(serial_bytes.data)()

        else:
            app = initial_app_loader()
            data = dill.dumps(initial_app_loader, recurse=True)
            serial_bytes = serial_utils.SerialBytes(data=data)
            serial_bytes_json = serial_bytes.model_dump()

        app_definition_json["app"] = app
        app_definition_json["initial_app_loader_dump"] = serial_bytes_json

        cls = pyschema_utils.WithClassInfo.get_class(app_definition_json)

        return cls.model_validate_json(app_definition_json)

    def jsonify_extra(self, content):
        # Called by jsonify for us to add any data we might want to add to the
        # serialization of `app`.
        if self.app_extra_json is not None:
            content["app"].update(self.app_extra_json)

        return content

    @staticmethod
    def _submit_feedback_functions(
        record: record_schema.Record,
        feedback_functions: Sequence[core_feedback.Feedback],
        connector: db_connector.DBConnector,
        app: Optional[AppDefinition] = None,
        on_done: Optional[
            Callable[
                [
                    Union[
                        feedback_schema.FeedbackResult,
                        Future[feedback_schema.FeedbackResult],
                    ]
                ],
                None,
            ]
        ] = None,
    ) -> List[
        Tuple[core_feedback.Feedback, Future[feedback_schema.FeedbackResult]]
    ]:
        """Schedules to run the given feedback functions.

        Args:
            record: The record on which to evaluate the feedback functions.

            feedback_functions: A collection of feedback functions to evaluate.

            connector: The database connector to use.

            app: The app that produced the given record. If not provided, it is
                looked up from the database of this `TruSession` instance

            on_done: A callback to call when each feedback function is done.

        Returns:

            List[Tuple[Feedback, Future[FeedbackResult]]]

            Produces a list of tuples where the first item in each tuple is the
            feedback function and the second is the future of the feedback result.
        """

        app_id = record.app_id

        if app is None:
            app = AppDefinition.model_validate(connector.get_app(app_id=app_id))
            if app is None:
                raise RuntimeError(
                    f"App {app_id} not present in db. "
                    "Either add it with `TruSession.add_app` or provide `app_json` to `TruSession.run_feedback_functions`."
                )

        else:
            assert (
                app_id == app.app_id
            ), "Record was produced by a different app."

            if connector.get_app(app_id=app.app_id) is None:
                logger.warning(
                    f"App {app_id} was not present in database. Adding it."
                )
                connector.add_app(app=app)

        feedbacks_and_futures = []

        tp = threading_utils.TP()

        for ffunc in feedback_functions:
            # Run feedback function and the on_done callback. This makes sure
            # that Future.result() returns only after on_done has finished.
            def run_and_call_callback(
                ffunc: core_feedback.Feedback,
                app: AppDefinition,
                record: record_schema.Record,
            ):
                token = None
                try:
                    token = core_instruments.do_not_track.set(True)
                    temp = ffunc.run(app=app, record=record)
                    if on_done is not None:
                        try:
                            on_done(temp)
                        finally:
                            return temp
                    return temp
                finally:
                    if token is not None:
                        core_instruments.do_not_track.reset(token)

            fut: Future[feedback_schema.FeedbackResult] = tp.submit(
                run_and_call_callback,
                ffunc=ffunc,
                app=app,
                record=record,
            )

            # Have to roll the on_done callback into the submitted function
            # because the result() is returned before callback runs otherwise.
            # We want to do db work before result is returned.

            feedbacks_and_futures.append((ffunc, fut))

        return feedbacks_and_futures

    @staticmethod
    def get_loadable_apps():
        """Gets a list of all of the loadable apps.

        Warning:
            This is an experimental feature with ongoing work.

        This is those that have `initial_app_loader_dump` set.
        """

        rets = []

        from trulens.core.session import TruSession

        session = TruSession()

        apps = session.get_apps()
        for app in apps:
            dump = app.get("initial_app_loader_dump")
            if dump is not None:
                rets.append(app)

        return rets

    def dict(self):
        # Unsure if the check below is needed. Sometimes we have an `app.App`` but
        # it is considered an `AppDefinition` and is thus using this definition
        # of `dict` instead of the one in `app.App`.
        from trulens.core.app import App

        if isinstance(self, App):
            return json_utils.jsonify(self, instrument=self.instrument)
        else:
            return json_utils.jsonify(self)

    @classmethod
    def select_inputs(cls) -> serial_utils.Lens:
        """Get the path to the main app's call inputs."""

        return select_schema.Select.RecordCall.args

    @classmethod
    def select_outputs(cls) -> serial_utils.Lens:
        """Get the path to the main app's call outputs."""

        return select_schema.Select.RecordCall.rets


# HACK013: Need these if using __future__.annotations .
AppDefinition.model_rebuild()
