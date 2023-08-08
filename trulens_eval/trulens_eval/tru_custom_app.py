"""
# Langchain instrumentation and monitoring.
"""

import logging
from pprint import PrettyPrinter
from typing import Any, Callable, ClassVar, Dict, Optional, Set

from pydantic import Field

from trulens_eval.app import App
from trulens_eval.instruments import Instrument
from trulens_eval.util import JSONPath
from trulens_eval.util import Class
from trulens_eval.util import FunctionOrMethod

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

from pydantic.fields import ModelField


class TruCustomApp(App):
    """
    Wrap a langchain Chain to capture its configuration and evaluation steps. 
    """

    app: Any

    # TODO: what if _acall is being used instead?
    root_callable: ClassVar[FunctionOrMethod] = Field(None)

    methods_to_instrument: ClassVar[Set[Callable]] = set(
        []
    )  # = Field(default_factory=set, exclude=True)

    @classmethod
    def instrument_method(cls, func):
        # cls.root_methods.add(func)

        cls.methods_to_instrument.add(func)

        return func

    # https://github.com/pydantic/pydantic/issues/1937
    @classmethod
    def add_fields(cls, **field_definitions: Any):
        new_fields: Dict[str, ModelField] = {}
        new_annotations: Dict[str, Optional[type]] = {}

        for f_name, f_def in field_definitions.items():
            if isinstance(f_def, tuple):
                try:
                    f_annotation, f_value = f_def
                except ValueError as e:
                    raise Exception(
                        'field definitions should either be a tuple of (<type>, <default>) or just a '
                        'default value, unfortunately this means tuples as '
                        'default values are not allowed'
                    ) from e
            else:
                f_annotation, f_value = None, f_def

            if f_annotation:
                new_annotations[f_name] = f_annotation

            new_fields[f_name] = ModelField.infer(
                name=f_name,
                value=f_value,
                annotation=f_annotation,
                class_validators=None,
                config=cls.__config__
            )

        cls.__fields__.update(new_fields)
        cls.__annotations__.update(new_annotations)

    # Normally pydantic does not like positional args but chain here is
    # important enough to make an exception.
    def __init__(self, app: Any, methods_to_instrument=None, **kwargs):
        """
        Wrap a langchain chain for monitoring.

        Arguments:
        - app: Any -- the custom app object being wrapped.
        - More args in App
        - More args in AppDefinition
        - More args in WithClassInfo
        """

        super().update_forward_refs()

        # TruChain specific:
        kwargs['app'] = app
        kwargs['root_class'] = Class.of_object(app)
        kwargs['instrument'] = Instrument(
            root_methods=set(
                [TruCustomApp.with_record, TruCustomApp.awith_record]
            ),
            on_new_record=self._on_new_record,
            on_add_record=self._on_add_record
        )

        super().__init__(**kwargs)

        #for m in self.root_methods:
        #    m_with_record = lambda inner_self, *args, **kwargs: TruCustomApp.with_record(inner_self, m, *args, **kwargs)
        #    m_with_record_name = m.__name__ + "_with_record"
        # self.__fields__[m_with_record_name] = Field(None, final=False)
        #    print(f"created method {m_with_record_name}")
        # TruCustomApp.add_fields(**{m_with_record_name: m_with_record})
        #    setattr(TruCustomApp, m_with_record_name, m_with_record)

        #print(self.__fields__)

        methods_to_instrument = methods_to_instrument or set([])

        for m, query in methods_to_instrument.items():
            setattr(
                m.__self__, m.__name__,
                self.instrument.instrument_tracked_method(
                    query=query,
                    func=m,
                    method_name=m.__name__,
                    cls=m.__self__.__class__,
                    obj=m.__self__
                )
            )

    def __getattr__(self, __name: str) -> Any:
        # A message for cases where a user calls something that the wrapped
        # app has but we do not wrap yet.

        if hasattr(self.app, __name):
            return RuntimeError(
                f"TruCustomApp has no attribute {__name} but the wrapped app ({type(self.app)}) does. ",
                f"If you are calling a {type(self.app)} method, retrieve it from that app instead of from `TruCustomApp`. "
            )
        else:
            raise RuntimeError(
                f"TruCustomApp nor wrapped app have attribute named {__name}."
            )
