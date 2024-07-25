"""
# _TruLens-Eval_ build script

To build:

```bash
python setup.py bdist_wheel
```

TODO: It is more standard to configure a lot of things we configure
here in a setup.cfg file instead. It is unclear whether we can do everything
with a config file though so we may need to keep this script or parts of it.
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
