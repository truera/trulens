"""
# Llama_index instrumentation and monitoring. 
"""

from collections import defaultdict
from datetime import datetime
from inspect import BoundArguments
from inspect import signature
from inspect import stack
import logging
import os
from pprint import PrettyPrinter
import threading as th
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel
from pydantic import Field

from trulens_eval.schema import FeedbackMode, Method, MethodIdent

from trulens_eval.schema import Record
from trulens_eval.schema import RecordChainCall
from trulens_eval.schema import RecordChainCallMethod
from trulens_eval.schema import Cost
from trulens_eval.tru_db import Query
from trulens_eval.tru_db import TruDB
from trulens_eval.tru_feedback import Feedback
from trulens_eval.tru import Tru
from trulens_eval.schema import FeedbackResult
from trulens_eval.schema import LlamaIndexModel
from trulens_eval.tru_model import TruModel
from trulens_eval.util import get_local_in_call_stack
from trulens_eval.util import TP, JSONPath, jsonify, noserio

import llama_index

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

class TruLlama(LlamaIndexModel, TruModel):
    """
    Wrap a langchain Chain to capture its configuration and evaluation steps. 
    """

    class Config:
        arbitrary_types_allowed = True

    # See LangChainModel for serializable fields.

    # Feedback functions to evaluate on each record.
    feedbacks: Sequence[Feedback] = Field(exclude=True)

    # Database interfaces for models/records/feedbacks.
    # NOTE: Maybe move to schema.Model .
    tru: Optional[Tru] = Field(exclude=True)

    # Database interfaces for models/records/feedbacks.
    # NOTE: Maybe mobe to schema.Model .
    db: Optional[TruDB] = Field(exclude=True)

    # TODO:

    def query(self, *args, **kwargs) -> llama_index.response.schema.Response:
        res, _ = self.query_with_record(*args, **kwargs)

        return res
    
    def query_with_record(self, *args, **kwargs) -> Tuple[llama_index.response.schema.Response, Record]:
        res = self.engine.query(*args, **kwargs)


