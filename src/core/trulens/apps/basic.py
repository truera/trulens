"""Basic input output instrumentation and monitoring."""

from inspect import BoundArguments
from inspect import Signature
from inspect import signature
import logging
from pprint import PrettyPrinter
from typing import Any, Callable, ClassVar, List, Optional

from pydantic import Field
from trulens.core import app as core_app
from trulens.core import instruments as core_instruments
from trulens.core.instruments import InstrumentedMethod
from trulens.core.utils import pyschema as pyschema_utils

logger = logging.getLogger(__name__)

pp = PrettyPrinter()


class TruWrapperApp:
    """Wrapper of basic apps.

    This will be wrapped by instrumentation.

    Warning:
        Because `TruWrapperApp` may wrap different types of callables, we cannot
        patch the signature to anything consistent. Because of this, the
        dashboard/record for this call will have `*args`, `**kwargs` instead of
        what the app actually uses. We also need to adjust the main_input lookup
        to get the correct signature. See note there.
    """

    def _call(self, *args, **kwargs):
        return self._call_fn(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def __init__(self, call_fn: Callable):
        self._call_fn = call_fn


class TruBasicCallableInstrument(core_instruments.Instrument):
    """Basic app instrumentation."""

    class Default:
        """Default instrumentation specification for basic apps."""

        CLASSES = lambda: {TruWrapperApp}

        # Instrument only methods with these names and of these classes.
        METHODS: List[InstrumentedMethod] = [
            InstrumentedMethod("_call", TruWrapperApp)
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(
            include_classes=TruBasicCallableInstrument.Default.CLASSES(),
            include_methods=TruBasicCallableInstrument.Default.METHODS,
            *args,
            **kwargs,
        )


class TruBasicApp(core_app.App):
    """Instantiates a Basic app that makes little assumptions.

    Assumes input text and output text.

    Example:
        ```python
        def custom_application(prompt: str) -> str:
            return "a response"

        from trulens.apps.basic import TruBasicApp
        # f_lang_match, f_qa_relevance, f_context_relevance are feedback functions
        tru_recorder = TruBasicApp(custom_application,
            app_name="Custom Application",
            app_version="1",
            feedbacks=[f_lang_match, f_qa_relevance, f_context_relevance])

        # Basic app works by turning your callable into an app
        # This app is accessible with the `app` attribute in the recorder
        with tru_recorder as recording:
            tru_recorder.app(question)

        tru_record = recording.records[0]
        ```

        See [Feedback
        Functions](https://www.trulens.org/trulens/api/feedback/) for
        instantiating feedback functions.

    Args:
        text_to_text: A str to str callable.

        app: A TruWrapperApp instance. If not provided, `text_to_text` must
            be provided.

        **kwargs: Additional arguments to pass to [App][trulens.core.app.App]
            and [AppDefinition][trulens.core.schema.app.AppDefinition]
    """

    app: TruWrapperApp
    """The app to be instrumented."""

    # TODEP
    root_callable: ClassVar[pyschema_utils.FunctionOrMethod] = Field(None)
    """The root callable to be instrumented.

    This is the method that will be called by the main_input method."""

    def __init__(
        self,
        text_to_text: Optional[Callable[[str], str]] = None,
        app: Optional[TruWrapperApp] = None,
        **kwargs: Any,
    ):
        if text_to_text is not None:
            app = TruWrapperApp(text_to_text)
        else:
            assert (
                app is not None
            ), "Need to provide either `app: TruWrapperApp` or a `text_to_text: Callable`."
        if "main_method" in kwargs and kwargs["main_method"] is not None:
            raise ValueError(
                "`main_method` should not be provided for `TruBasicApp`!"
            )

        kwargs["app"] = app
        kwargs["root_class"] = pyschema_utils.Class.of_object(app)
        kwargs["instrument"] = TruBasicCallableInstrument(app=self)
        kwargs["main_method"] = app._call

        super().__init__(**kwargs)

    def main_call(self, human: str) -> str:
        # If available, a single text to a single text invocation of this app.

        return self.app._call(human)

    def main_input(
        self, func: Callable, sig: Signature, bindings: BoundArguments
    ) -> str:
        if hasattr(
            TruWrapperApp._call, core_instruments.Instrument.INSTRUMENT
        ) and func == getattr(
            TruWrapperApp._call, core_instruments.Instrument.INSTRUMENT
        ):
            # If func is the wrapper app _call, replace the signature and
            # bindings based on the actual containing callable instead of
            # self.app._call . This needs to be done since the a TruWrapperApp
            # may be wrapping apps with different signatures on their callables
            # so TruWrapperApp._call cannot have a consistent signature
            # statically. Note also we are looking up the Instrument.INSTRUMENT
            # attribute here since the method is instrumented and overridden by
            # another wrapper in the process with the original accessible at
            # this attribute.

            sig = signature(self.app._call_fn)
            # Skipping self as TruWrapperApp._call takes in self, but
            # self.app._call_fn does not.
            bindings = sig.bind(*bindings.args[1:], **bindings.kwargs)

        return super().main_input(func, sig, bindings)

    def call_with_record(self, *args, **kwargs) -> None:
        self._throw_dep_message(method="call", is_async=False, with_record=True)


TruBasicApp.model_rebuild()
