from typing import Optional

from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.serial import SerialModel


class Provider(SerialModel, WithClassInfo):

    class Config:
        arbitrary_types_allowed = True

    endpoint: Optional[Endpoint]

    def __init__(self, name: str = None, **kwargs):
        # for WithClassInfo:
        kwargs['obj'] = self

        super().__init__(name=name, **kwargs)
