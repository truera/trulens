"""
Read secrets from .env for exporting to python scripts. Usage:
```python
    from keys import *
```
Will get you access to all of the vars defined in .env in wherever you put that import statement.
"""

import os

import cohere
import dotenv

config = dotenv.dotenv_values(".env")

for k, v in config.items():
    print(f"KEY SET: {k}")
    globals()[k] = v

    # set them into environment as well
    os.environ[k] = v

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
