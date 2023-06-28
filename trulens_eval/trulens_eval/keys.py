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

import cohere
import dotenv
from trulens_eval.util import UNICODE_CHECK, UNICODE_STOP

from trulens_eval.util import caller_frame

logger = logging.getLogger(__name__)


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
        # print(f"{config_file}: {k}")
        globals()[k] = v

        # set them into environment as well
        os.environ[k] = v


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

    set_openai_key()
