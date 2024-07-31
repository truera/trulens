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
from setuptools.command.build import build
from setuptools.command.develop import develop


def build_record_viewer():
    print("running npm i")
    os.system("npm i --prefix trulens/dashboard/react_components/record_viewer")
    print("running npm run build")
    os.system(
        "npm run --prefix trulens/dashboard/react_components/record_viewer build"
    )


class BuildJavascript(build):
    def run(self):
        """Custom build command to run npm commands before building the package.

        This builds the record timeline component for the dashboard.
        """
        build_record_viewer()
        build.run(self)


class DevelopJavascript(develop):
    def run(self):
        """Custom develop command to run npm commands before installing the package in develop mode (-e).

        This builds the record timeline component for the dashboard.
        """
        build_record_viewer()
        develop.run(self)


setup(
    cmdclass={"build": BuildJavascript, "develop": DevelopJavascript},
)
