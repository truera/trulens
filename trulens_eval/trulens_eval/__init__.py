"""
# Trulens-eval LLM Evaluation Library

This top-level import should include everything to get started.

## Module organization/dependency

Modules on lower lines should not import modules on same or above lines as
otherwise you might get circular import errors.

- `__init__.py`

- all UI/dashboard components

- `tru_chain.py` `tru_custom_app.py`

- `tru_llama.py` (note: llama_index uses langchain internally for some things)

- `tru.py`

- `utils`

    - `llama.py` `langchain.py` `trulens.py`

- `feedback`

    - `__init__.py`

    - `provider`

        - `__init__.py`

        - `endpoint`

            - `__init__.py`

            - `openai.py` `hugs.py`

            - `base.py` 

        - `hugs.py` `openai.py` `cohere.py`

        - `base.py`

    - `groundedness.py` `groundtruth.py`

    - `feedback.py` `prompts.py`

- `tru_basic_app.py` TODO: bad placement

- `app.py`

- `db.py`

- `instruments.py`

- `schema.py`

- `utils`

    - `json.py`

- `keys.py`

- `utils`

    - `pyschema.py`

    - `threading.py` `serial.py`

    - `python.py` `text.py` `generated.py` `containers.py` `imports.py`

TO PLACE

`utils/command_line.py`
`utils/notebook_utils.py`
`utils/__init__.py`

"""

__version__ = "0.18.1"

from trulens_eval.feedback import Bedrock
from trulens_eval.feedback import Feedback
from trulens_eval.feedback import Huggingface
from trulens_eval.feedback import LiteLLM
from trulens_eval.feedback import OpenAI
from trulens_eval.feedback.provider import Provider
from trulens_eval.schema import FeedbackMode
from trulens_eval.schema import Query
from trulens_eval.schema import Select
from trulens_eval.tru import Tru
from trulens_eval.tru_basic_app import TruBasicApp
from trulens_eval.tru_chain import TruChain
from trulens_eval.tru_custom_app import instrument
from trulens_eval.tru_custom_app import TruCustomApp
from trulens_eval.tru_llama import TruLlama
from trulens_eval.utils.threading import TP

__all__ = [
    'Tru',
    'TruBasicApp',
    'TruCustomApp',
    'TruChain',
    'TruLlama',
    'Feedback',
    'OpenAI',
    'LiteLLM',
    'Bedrock',
    'Huggingface',
    'FeedbackMode',
    'Provider',
    'Query',  # to deprecate in 0.3.0
    'Select',
    'TP'
]
