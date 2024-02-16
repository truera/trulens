"""
# Trulens-eval LLM Evaluation Library

This top-level import should include everything to get started.

## Module organization/dependency

Modules on lower lines should not import modules on same or above lines as
otherwise you might get circular import errors.

- `tests/`

- `db_migration.py`

- `__init__.py`

- all UI/dashboard components, `pages`, `react_components`, `ux`, `Leaderboard.py`, `appui.py`

- `appui.py`

- `tru_chain.py` `tru_custom_app.py` `tru_virtual.py`

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

            - `openai.py` `hugs.py` `bedrock.py` `litellm.py` `langchain.py`

            - `base.py` 

        - `hugs.py` `openai.py` `bedrock.py` `litellm.py` `langchain.py`

        - `base.py`

    - `groundedness.py` `groundtruth.py` `prompts.py` `embeddings.py`

    - `feedback.py`

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

`database/`
`feedback/v2/`
`utils/command_line.py`
`utils/notebook_utils.py`
`utils/asynchro.py`
`utils/text.py`
`utils/__init__.py`

"""

__version__ = "0.23.0"

from trulens_eval.feedback import Feedback
from trulens_eval.feedback import Langchain
from trulens_eval.feedback.provider import Provider
from trulens_eval.feedback.provider.hugs import Huggingface
from trulens_eval.schema import FeedbackMode
from trulens_eval.schema import Select
from trulens_eval.tru import Tru
from trulens_eval.tru_basic_app import TruBasicApp
from trulens_eval.tru_chain import TruChain
from trulens_eval.tru_custom_app import TruCustomApp
from trulens_eval.tru_virtual import TruVirtual
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_BEDROCK
from trulens_eval.utils.imports import REQUIREMENT_LITELLM
from trulens_eval.utils.imports import REQUIREMENT_LLAMA
from trulens_eval.utils.imports import REQUIREMENT_OPENAI
from trulens_eval.utils.threading import TP

with OptionalImports(messages=REQUIREMENT_LLAMA):
    from trulens_eval.tru_llama import TruLlama

with OptionalImports(messages=REQUIREMENT_LITELLM):
    from trulens_eval.feedback.provider.litellm import LiteLLM

with OptionalImports(messages=REQUIREMENT_BEDROCK):
    from trulens_eval.feedback.provider.bedrock import Bedrock

with OptionalImports(messages=REQUIREMENT_OPENAI):
    from trulens_eval.feedback.provider.openai import AzureOpenAI
    from trulens_eval.feedback.provider.openai import OpenAI

__all__ = [
    "Tru",  # main interface

    # app types
    "TruBasicApp",
    "TruCustomApp",
    "TruChain",
    "TruLlama",
    "TruVirtual",

    # app setup
    "FeedbackMode",

    # feedback setup
    "Feedback",
    "Select",

    # feedback providers
    "Provider",
    "AzureOpenAI",
    "OpenAI",
    "Langchain",
    "LiteLLM",
    "Bedrock",
    "Huggingface",

    # misc utility
    "TP",
]
