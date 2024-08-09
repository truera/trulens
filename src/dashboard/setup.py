"""
# trulens-dashboard build script

To build:

```bash
cd src/dashboard
python -m build
```
"""

import os

from setuptools import setup


def build_record_viewer():
    if os.path.exists("react_components/record_viewer"):
        print("running npm i")
        os.system("npm i --prefix react_components/record_viewer")
        print("running npm run build")
        os.system("npm run --prefix react_components/record_viewer build")


if __name__ == "__main__":
    build_record_viewer()
    setup()
