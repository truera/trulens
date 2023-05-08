"""
Read secrets from .env for exporting to python scripts. Usage:

```python
    from keys import *
```

Will get you access to all of the vars defined in .env in wherever you put that import statement.
"""

import os

import dotenv

config = dotenv.dotenv_values(".env")

for k, v in config.items():
    print(f"got {k}")
    globals()[k] = v

    # set them into environment as well
    os.environ[k] = v

HUGGINGFACE_HEADERS = {
    "Authorization": f"Bearer {config['HUGGINGFACE_API_KEY']}"
}
