"""
Read secrets from .env for exporting to python scripts. Usage:

```python
from keys import *
```

Will get you access to all of the vars defined in .env in wherever you put that
import statement.
"""

import logging
import os
from pathlib import Path
import re
from typing import Any, Optional, Set, Union

import cohere
import dotenv

from trulens_eval.utils.python import caller_frame
from trulens_eval.utils.text import UNICODE_CHECK
from trulens_eval.utils.text import UNICODE_STOP

logger = logging.getLogger(__name__)

# Keep track of values that should not be shown in UI (or added to DB). This set
# is only used for cases where the name/key for a field is not useful to
# determine whether it should be redacted.
values_to_redact = set()

# Regex of keys (into dict/json) that should be redacted.
RE_KEY_TO_REDACT = re.compile('|'.join([
    r'.*api_key', # covers OpenAI class key 'api_key' and env vars ending in 'API_KEY'
    r'KAGGLE_KEY',
    r'SLACK_(TOKEN|SIGNING_SECRET)', # covers slack-related keys
    ]), re.IGNORECASE
)
# Env vars not covered as they are assumed non-sensitive:
# - PINECONE_ENV, e.g. "us-west1-gcp-free"
# - KAGGLE_USER

# TODO: Some method for letting users add more things to redact.

# The replacement value for redacted values.
REDACTED_VALUE = "__tru_redacted"

def should_redact_key(k: Optional[str]) -> bool:
    return isinstance(k, str) and RE_KEY_TO_REDACT.fullmatch(k)

def should_redact_value(v: Union[Any, str]) -> bool:
    return isinstance(v, str) and v in values_to_redact

def redact_value(v: Union[str, Any], k: Optional[str] = None) -> Union[str, Any]:
    """
    Determine whether the given value `v` should be redact it and redact it if
    so. If its key `k` (in a dict/json-like) is given, uses the key name to
    determine whether redaction is appropriate. If key `k` is not given, only
    redacts if `v` is a string and identical to one of the keys ingested using
    `setup_keys`.
    """

    if should_redact_key(k) or should_redact_value(v):
        return REDACTED_VALUE
    else:
        return v


def get_config():
    for path in [Path().cwd(), *Path.cwd().parents]:
        file = path / ".env"
        if file.exists():
            # print(f"Using {file}")

            return file

    return None


config_file = get_config()
if config_file is None:
    logger.warning(
        f"No .env found in {Path.cwd()} or its parents. "
        "You may need to specify secret keys in another manner."
    )

else:
    config = dotenv.dotenv_values(config_file)

    for k, v in config.items():
        globals()[k] = v

        # Set them into environment as well
        os.environ[k] = v

        # Put value in redaction list.
        values_to_redact.add(v)


def set_openai_key():
    if 'OPENAI_API_KEY' in os.environ:
        import openai
        openai.api_key = os.environ["OPENAI_API_KEY"]


global cohere_agent
cohere_agent = None


def get_cohere_agent():
    global cohere_agent
    if cohere_agent is None:
        cohere.api_key = os.environ['COHERE_API_KEY']
        cohere_agent = cohere.Client(cohere.api_key)
    return cohere_agent


def get_huggingface_headers():
    HUGGINGFACE_HEADERS = {
        "Authorization": f"Bearer {os.environ['HUGGINGFACE_API_KEY']}"
    }
    return HUGGINGFACE_HEADERS


def setup_keys(**kwargs):
    global config_file

    config_file = get_config()
    if config_file is None:
        logger.warning(
            f"No .env found in {Path.cwd()} or its parents. "
            "You may need to specify secret keys in another manner."
        )

    to_global = dict()

    globs = caller_frame(offset=1).f_globals

    for k, v in kwargs.items():
        if v is not None and "fill" not in v.lower():
            to_global[k] = v
            print(f"{UNICODE_CHECK} Key {k} set explicitly.")
            continue

        if k in globs:
            print(f"{UNICODE_CHECK} Key {k} was already set.")
            continue

        if k in os.environ:
            v = os.environ[k]
            to_global[k] = v
            print(f"{UNICODE_CHECK} Key {k} set from environment.")
            continue

        if config_file is not None:
            if k in config:
                v = config[k]
                print(f"{UNICODE_CHECK} Key {k} set from {config_file} .")
                to_global[k] = v
                continue

        if "fill" in v:
            raise RuntimeError(
                f"""{UNICODE_STOP} Key {k} needs to be set; please provide it in one of these ways:
- in a variable {k} prior to this check, 
- in your variable environment, 
- in a .env file in {Path.cwd()} or its parents, or 
- explicitly passed to `setup_keys`.
"""
            )

    for k, v in to_global.items():
        globs[k] = v
        os.environ[k] = v

        values_to_redact.add(v)

    set_openai_key()
