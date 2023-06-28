"""
# API keys and configuration 

## Setting keys

Most example notebooks come with a key setup line like this:

```python 
from trulens_eval.keys import setup_keys

setup_keys(
    OPENAI_API_KEY="to fill in", 
    HUGGINGFACE_API_KEY="to fill in"
)
```

This line checks that you have the requisite api keys set before continuing the
notebook. They do not need to be provided, however, right on this line. There
are several ways to make sure this check passes:

- *Explicit* -- Explicitly provide key values to `setup_keys`.

- *Python* -- Define variables before this check like this:

```python
OPENAI_API_KEY="something"
```

- *Environment* -- Set them in your environment variable. They should be visible when you execute:

```python
import os
print(os.environ)
```

- *.env* -- Set them in a .env file in the same folder as the example notebook or one of
  its parent folders. An example of a .env file is found in
  `trulens_eval/trulens_eval/env.example` .

- *3rd party* -- For some keys, set them as arguments to the 3rd-party endpoint class. For
  example, with `openai`, do this ahead of the `setup_keys` check:

```python
import openai
openai.api_key = "something"
```

- *Endpoint class* For some keys, set them as arguments to trulens_eval endpoint class that
  manages the endpoint. For example, with `openai`, do this ahead of the
  `setup_keys` check:

```python
from trulens_eval.provider_apis import OpenAIEndpoint
openai_endpoint = OpenAIEndpoint(api_key="something")
```

- *Provider class* For some keys, set them as arguments to trulens_eval feedback
  collection ("provider") class that makes use of the relevant endpoint. For
  example, with `openai`, do this ahead of the `setup_keys` check:

```python
from trulens_eval.feedback import OpenAI
openai_feedbacks = OpenAI(api_key="something")
```

In the last two cases, please note that the settings are global. Even if you
create multiple OpenAI or OpenAIEndpoint objects, they will share the
configuration of keys (and other openai attributes).

## Other API attributes

Some providers may require additional configuration attributes beyond api key.
For example, `openai` usage via azure require special keys. To set those, you
should use the 3rd party class method of configuration. For example with
`openai`:

```python
import openai

openai.api_type = "azure"
openai.api_key = "..."
openai.api_base = "https://example-endpoint.openai.azure.com"
openai.api_version = "2023-05-15"  # subject to change
# See https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/switching-endpoints .
```

Our example notebooks will only check that the api_key is set but will make use
of the configured openai object as needed to compute feedback.
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
    """
    Check various sources of api configuration values like secret keys and set
    env variables for each of them. We use env variables as the canonical
    storage of these keys, regardless of how they were specified.
    """

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
            print(f"{UNICODE_CHECK} Variable {k} set explicitly.")
            continue

        if k in globs:
            print(f"{UNICODE_CHECK} Variable {k} was already set.")
            continue

        if k in os.environ:
            # Note this option also applies to cases where we copy the key from
            # a specific class like openai as that process sets the relevant env
            # variable as well so we can access it here.
            v = os.environ[k]
            to_global[k] = v
            print(f"{UNICODE_CHECK} Variable {k} set from environment.")
            continue

        if config_file is not None:
            if k in config:
                v = config[k]
                print(f"{UNICODE_CHECK} Variable {k} set from {config_file} .")
                to_global[k] = v
                continue

        if "fill" in v:
            raise RuntimeError(\
f"""{UNICODE_STOP} Variable {k} needs to be set; please provide it in one of these ways:

  - in a variable {k} prior to this check, 
  - in your variable environment, 
  - in a .env file in {Path.cwd()} or its parents,
  - explicitly passed to `setup_keys`,
  - passed to the endpoint or feedback collection constructor that needs it (`trulens_eval.provider_apis.OpenAIEndpoint`, etc.), or
  - set in api utility class that expects it (i.e. `openai`, etc.).

For the last two options, the name of the argument may differ from {k} (i.e. `openai.api_key` for `OPENAI_API_KEY`).
""")

    for k, v in to_global.items():
        globs[k] = v
        os.environ[k] = v

    set_openai_key()