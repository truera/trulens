from distutils import log  # DEP
from distutils.command.build import build  # DEP
import os
from pathlib import Path

from pip._internal.req import parse_requirements
from setuptools import find_namespace_packages
from setuptools import setup

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


class javascript_build(build):

    def run(self):
        log.info("running npm i")
        os.system("npm i --prefix trulens_eval/react_components/record_viewer")
        log.info("running npm run build")
        os.system(
            "npm run --prefix trulens_eval/react_components/record_viewer build"
        )
        build.run(self)


setup(
    name="trulens_eval",
    cmdclass={
        'build': javascript_build,
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
