"""
# trulens-dashboard build script

To build:

```bash
cd src/dashboard
python -m build
```
"""

import os
import subprocess

from setuptools import setup


def build_record_viewer():
    if os.path.exists("react_components/record_viewer"):
        try:
            print("running npm i")
            subprocess.check_call(
                [
                    "npm",
                    "i",
                    "--prefix",
                    "react_components/record_viewer",
                ]
            )
        except FileNotFoundError:
            print("`npm` is not installed. Skipping build")
        print("running npm run build")
        subprocess.check_call(
            [
                "npm",
                "run",
                "--prefix",
                "react_components/record_viewer",
                "build",
            ]
        )


if __name__ == "__main__":
    build_record_viewer()
    setup()
