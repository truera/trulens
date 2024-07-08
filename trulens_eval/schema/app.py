"""Serializable app-related classes."""

from __future__ import annotations

from enum import Enum
import logging
from typing import Any, Callable, ClassVar, Optional, Sequence, Type

import dill
import humanize

from trulens_eval import app as mod_app
from trulens_eval.schema import base as mod_base_schema
from trulens_eval.schema import feedback as mod_feedback_schema
from trulens_eval.schema import types as mod_types_schema
from trulens_eval.utils import pyschema
from trulens_eval.utils import serial
from trulens_eval.utils.json import jsonify
from trulens_eval.utils.json import obj_id_of_obj

logger = logging.getLogger(__name__)


class AppDefinition(pyschema.WithClassInfo, serial.SerialModel):
    """Serialized fields of an app here whereas [App][trulens_eval.app.App]
    contains non-serialized fields."""

    app_id: mod_types_schema.AppID  # str
    """Unique identifier for this app."""

    tags: mod_types_schema.Tags  # str
    """Tags for the app."""

    metadata: mod_types_schema.Metadata  # dict  # TODO: rename to meta for consistency with other metas
    """Metadata for the app."""

    feedback_definitions: Sequence[mod_feedback_schema.FeedbackDefinition] = []
    """Feedback functions to evaluate on each record."""

    feedback_mode: mod_feedback_schema.FeedbackMode = mod_feedback_schema.FeedbackMode.WITH_APP_THREAD
    """How to evaluate feedback functions upon producing a record."""

    root_class: pyschema.Class
    """Class of the main instrumented object.
    
    Ideally this would be a [ClassVar][typing.ClassVar] but since we want to check this without
    instantiating the subclass of
    [AppDefinition][trulens_eval.schema.app.AppDefinition] that would define it, we
    cannot use [ClassVar][typing.ClassVar].
    """

    root_callable: ClassVar[pyschema.FunctionOrMethod]
    """App's main method. 
    
    This is to be filled in by subclass.
    """

    app: serial.JSONized[AppDefinition]
    """Wrapped app in jsonized form."""

    initial_app_loader_dump: Optional[serial.SerialBytes] = None
    """Serialization of a function that loads an app.

    Dump is of the initial app state before any invocations. This can be used to
    create a new session.

    Warning:
        Experimental work in progress.
    """

    app_extra_json: serial.JSON
    """Info to store about the app and to display in dashboard. 
    
    This can be used even if app itself cannot be serialized. `app_extra_json`,
    then, can stand in place for whatever data the user might want to keep track
    of about the app.
    """

    def __init__(
        self,
        app_id: Optional[mod_types_schema.AppID] = None,
        tags: Optional[mod_types_schema.Tags] = None,
        metadata: Optional[mod_types_schema.Metadata] = None,
        feedback_mode: mod_feedback_schema.FeedbackMode = mod_feedback_schema.
        FeedbackMode.WITH_APP_THREAD,
        app_extra_json: serial.JSON = None,
        **kwargs
    ):

        # for us:
        kwargs['app_id'] = "temporary"  # will be adjusted below
        kwargs['feedback_mode'] = feedback_mode
        kwargs['tags'] = ""
        kwargs['metadata'] = {}
        kwargs['app_extra_json'] = app_extra_json or dict()

        super().__init__(**kwargs)

        if app_id is None:
            app_id = obj_id_of_obj(obj=self.model_dump(), prefix="app")

        self.app_id = app_id

        if tags is None:
            tags = "-"  # Set tags to a "-" if None is provided
        self.tags = tags

        if metadata is None:
            metadata = {}
        self.metadata = metadata

        # EXPERIMENTAL
        if 'initial_app_loader' in kwargs:
            try:
                dump = dill.dumps(kwargs['initial_app_loader'], recurse=True)

                if len(dump) > mod_base_schema.MAX_DILL_SIZE:
                    logger.warning(
                        "`initial_app_loader` dump is too big (%s) > %s bytes). "
                        "If you are loading large objects, include the loading logic inside `initial_app_loader`.",
                        humanize.naturalsize(len(dump)),
                        humanize.naturalsize(mod_base_schema.MAX_DILL_SIZE)
                    )
                else:
                    self.initial_app_loader_dump = serial.SerialBytes(data=dump)

                    # This is an older serialization approach that saved things
                    # in local files instead of the DB. Leaving here for now as
                    # serialization of large apps might make this necessary
                    # again.
                    """
                    path_json = Path.cwd() / f"{app_id}.json"
                    path_dill = Path.cwd() / f"{app_id}.dill"

                    with path_json.open("w") as fh:
                        fh.write(json_str_of_obj(self))

                    with path_dill.open("wb") as fh:
                        fh.write(dump)

                    print(f"Wrote loadable app to {path_json} and {path_dill}.")
                    """

            except Exception as e:
                logger.warning(
                    "Could not serialize app loader. "
                    "Some trulens features may not be available: %s", e
                )

    @staticmethod
    def continue_session(
        app_definition_json: serial.JSON, app: Any
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

        app_definition_json['app'] = app

        cls = pyschema.WithClassInfo.get_class(app_definition_json)

        return cls(**app_definition_json)

    @staticmethod
    def new_session(
        app_definition_json: serial.JSON,
        initial_app_loader: Optional[Callable] = None
    ) -> AppDefinition:
        """Create an app instance at the start of a session.
        
        Warning:
            This is an experimental feature with ongoing work.

        Create a copy of the json serialized app with the enclosed app being
        initialized to its initial state before any records are produced (i.e.
        blank memory).
        """

        serial_bytes_json: Optional[
            serial.JSON] = app_definition_json['initial_app_loader_dump']

        if initial_app_loader is None:
            assert serial_bytes_json is not None, "Cannot create new session without `initial_app_loader`."

            serial_bytes = serial.SerialBytes.model_validate(serial_bytes_json)

            app = dill.loads(serial_bytes.data)()

        else:
            app = initial_app_loader()
            data = dill.dumps(initial_app_loader, recurse=True)
            serial_bytes = serial.SerialBytes(data=data)
            serial_bytes_json = serial_bytes.model_dump()

        app_definition_json['app'] = app
        app_definition_json['initial_app_loader_dump'] = serial_bytes_json

        cls: Type[mod_app.App
                 ] = pyschema.WithClassInfo.get_class(app_definition_json)

        return cls.model_validate_json(app_definition_json)

    def jsonify_extra(self, content):
        # Called by jsonify for us to add any data we might want to add to the
        # serialization of `app`.
        if self.app_extra_json is not None:
            content['app'].update(self.app_extra_json)

        return content

    @staticmethod
    def get_loadable_apps():
        """Gets a list of all of the loadable apps.
        
        Warning:
            This is an experimental feature with ongoing work.

        This is those that have `initial_app_loader_dump` set.
        """

        rets = []

        from trulens_eval import Tru

        tru = Tru()

        apps = tru.get_apps()
        for app in apps:
            dump = app.get('initial_app_loader_dump')
            if dump is not None:
                rets.append(app)

        return rets

    def dict(self):
        # Unsure if the check below is needed. Sometimes we have an `app.App`` but
        # it is considered an `AppDefinition` and is thus using this definition
        # of `dict` instead of the one in `app.App`.

        if isinstance(self, mod_app.App):
            return jsonify(self, instrument=self.instrument)
        else:
            return jsonify(self)

    @classmethod
    def select_inputs(cls) -> serial.Lens:
        """Get the path to the main app's call inputs."""

        return getattr(
            mod_feedback_schema.Select.RecordCalls,
            cls.root_callable.default_factory().name
        ).args

    @classmethod
    def select_outputs(cls) -> serial.Lens:
        """Get the path to the main app's call outputs."""

        return getattr(
            mod_feedback_schema.Select.RecordCalls,
            cls.root_callable.default_factory().name
        ).rets


# HACK013: Need these if using __future__.annotations .
AppDefinition.model_rebuild()
