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

from pip._internal.req import parse_requirements
from setuptools import find_namespace_packages
from setuptools import setup
from setuptools.command.build import build
from setuptools.logging import logging

required_packages = list(
    map(
        lambda pip_req: str(pip_req.requirement),
        parse_requirements("trulens_eval/requirements.txt", session=None)
    )
)
optional_packages = list(
    map(
        lambda pip_req: str(pip_req.requirement),
        parse_requirements(
            "trulens_eval/requirements.optional.txt", session=None
        )
    )
)

class BuildJavascript(build):
    def run(self):
        """Custom build command to run npm commands before building the package.
    
        This builds the record timeline component for the dashboard.
        """

        logging.info("running npm i")
        os.system("npm i --prefix trulens_eval/react_components/record_viewer")
        logging.info("running npm run build")
        os.system(
            "npm run --prefix trulens_eval/react_components/record_viewer build"
        )
        build.run(self)


setup(
    name="trulens_eval",
    cmdclass={
        'build': BuildJavascript,
    },
    include_package_data=True,  # includes things specified in MANIFEST.in
    packages=find_namespace_packages(
        include=["trulens_eval", "trulens_eval.*"]
    ),
    python_requires='>= 3.8, < 3.13',
    entry_points={
        'console_scripts': [
            'trulens-eval=trulens_eval.utils.command_line:main'
        ],
    },
    install_requires=required_packages
)
