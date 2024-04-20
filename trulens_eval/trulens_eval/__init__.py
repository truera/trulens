"""
# Trulens-eval LLM Evaluation Library

This top-level import should include everything to get started.

## Module organization/dependency

Modules on lower lines should not import modules on same or above lines as
otherwise you might get circular import errors.

- `tests/`

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
- `database/`
- `feedback/v2/`
- `utils/command_line.py`
- `utils/notebook_utils.py`
- `utils/asynchro.py`
- `utils/text.py`
- `utils/__init__.py`

"""

__version_info__ = (0, 28, 0)
__version__ = '.'.join(map(str, __version_info__))

# This check is intentionally done ahead of the other imports as we want to
# print out a nice warning/error before an import error happens further down
# this sequence.
from trulens_eval.utils.imports import check_imports

check_imports()

from trulens_eval import tru as mod_tru
from trulens_eval import tru_basic_app as mod_tru_basic_app
from trulens_eval import tru_chain as mod_tru_chain
from trulens_eval import tru_custom_app as mod_tru_custom_app
from trulens_eval import tru_virtual as mod_tru_virtual
from trulens_eval.feedback import feedback as mod_feedback
from trulens_eval.feedback.provider import base as mod_provider
from trulens_eval.feedback.provider import hugs as mod_hugs_provider
from trulens_eval.feedback.provider import langchain as mod_langchain_provider
from trulens_eval.schema import app as mod_app_schema
from trulens_eval.schema import feedback as mod_feedback_schema
from trulens_eval.utils import imports as mod_imports_utils
from trulens_eval.utils import threading as mod_threading_utils

# Optional provider types.

with mod_imports_utils.OptionalImports(messages=mod_imports_utils.REQUIREMENT_LITELLM):
    from trulens_eval.feedback.provider.litellm import LiteLLM

with mod_imports_utils.OptionalImports(messages=mod_imports_utils.REQUIREMENT_BEDROCK):
    from trulens_eval.feedback.provider.bedrock import Bedrock

with mod_imports_utils.OptionalImports(messages=mod_imports_utils.REQUIREMENT_OPENAI):
    from trulens_eval.feedback.provider.openai import AzureOpenAI
    from trulens_eval.feedback.provider.openai import OpenAI

# Optional app types.

with mod_imports_utils.OptionalImports(messages=mod_imports_utils.REQUIREMENT_LLAMA):
    from trulens_eval.tru_llama import TruLlama

with mod_imports_utils.OptionalImports(messages=mod_imports_utils.REQUIREMENT_RAILS):
    from trulens_eval.tru_rails import TruRails

Tru = mod_tru.Tru
TruBasicApp = mod_tru_basic_app.TruBasicApp
TruChain = mod_tru_chain.TruChain
TruCustomApp = mod_tru_custom_app.TruCustomApp
TruVirtual = mod_tru_virtual.TruVirtual
TP = mod_threading_utils.TP
Feedback = mod_feedback.Feedback
Provider = mod_provider.Provider
Huggingface = mod_hugs_provider.Huggingface
Langchain = mod_langchain_provider.Langchain
FeedbackMode = mod_app_schema.FeedbackMode
Select = mod_feedback_schema.Select

__all__ = [
    "Tru",  # main interface

    # app types
    "TruBasicApp",
    "TruCustomApp",
    "TruChain",
    "TruLlama",
    "TruVirtual",
    "TruRails",

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
