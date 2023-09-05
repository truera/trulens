"""
# Basic input output instrumentation and monitoring.
"""

from inspect import BoundArguments
from inspect import Signature
import logging
from pprint import PrettyPrinter
from typing import Callable, ClassVar

from pydantic import Field

from trulens_eval.app import App
from trulens_eval.instruments import Instrument
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.pyschema import FunctionOrMethod

logger = logging.getLogger(__name__)

pp = PrettyPrinter()


class TruBasicCallableInstrument(Instrument):

    class Default:
        CLASSES = lambda: {TruWrapperApp}

        # Instrument only methods with these names and of these classes.
        METHODS = {"_call": lambda o: isinstance(o, TruWrapperApp)}

    def __init__(self, *args, **kwargs):
        super().__init__(
            include_classes=TruBasicCallableInstrument.Default.CLASSES(),
            include_methods=TruBasicCallableInstrument.Default.METHODS,
            *args,
            **kwargs
        )


class TruWrapperApp(object):
    # the class level call (Should be immutable from the __init__)
    _call: Callable = lambda self, *args, **kwargs: self._call_fn(
        *args, **kwargs
    )

    def __init__(self, call_fn: Callable):
        self._call_fn = call_fn


class TruBasicApp(App):
    """
    A Basic app that makes little assumptions. Assumes input text and output text 
    """

    app: TruWrapperApp

    root_callable: ClassVar[FunctionOrMethod] = Field(
        default_factory=lambda: FunctionOrMethod.
        of_callable(TruWrapperApp._call),
        const=True
    )

    def __init__(self, text_to_text: Callable, **kwargs):
        """
        Wrap a callable for monitoring.

        Arguments:
        - text_to_text: A callable string to string
        - More args in App
        - More args in AppDefinition
        - More args in WithClassInfo
        """
        assert isinstance(text_to_text("This should return a string"), str)
        super().update_forward_refs()
        app = TruWrapperApp(text_to_text)
        kwargs['app'] = TruWrapperApp(text_to_text)
        kwargs['root_class'] = Class.of_object(app)
        kwargs['instrument'] = TruBasicCallableInstrument(app=self)

        super().__init__(**kwargs)

        # Setup the DB-related things:
        self.post_init()

    def main_input(
        self, func: Callable, sig: Signature, bindings: BoundArguments
    ) -> str:
        if "input" in bindings.arguments:
            return bindings.arguments['input']

        return super().main_input(func, sig, bindings)

    def call_with_record(self, input: str, **kwargs):
        """ Run the callable and pass any kwargs.

        Returns:
            dict: record metadata
        """

        self._with_dep_message(method="_call", is_async=False, with_record=True)

        return self.with_record(self.app._call, input, **kwargs)
