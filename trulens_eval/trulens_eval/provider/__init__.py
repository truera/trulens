from typing import Optional
from trulens_eval.provider.endpoint import Endpoint
from trulens_eval.util import SerialModel, WithClassInfo


class Provider(SerialModel, WithClassInfo):

    class Config:
        arbitrary_types_allowed = True

    endpoint: Optional[Endpoint]

    def __init__(self, *args, **kwargs):
        # for WithClassInfo:
        kwargs['obj'] = self

        super().__init__(*args, **kwargs)

