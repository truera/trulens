import logging
from typing import Optional

from trulens_eval.feedback.provider.endpoint.endpoint import Endpoint
from trulens_eval.feedback.provider.hugs import Huggingface
from trulens_eval.feedback.provider.openai import OpenAI
from trulens_eval.util import SerialModel
from trulens_eval.util import WithClassInfo

logger = logging.getLogger(__name__)


class Provider(SerialModel, WithClassInfo):

    class Config:
        arbitrary_types_allowed = True

    endpoint: Optional[Endpoint]

    def __init__(self, *args, **kwargs):
        # for WithClassInfo:
        kwargs['obj'] = self

        super().__init__(*args, **kwargs)


__all__ = ['Provider', 'OpenAI', 'Huggingface']