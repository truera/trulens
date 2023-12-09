from distutils import log
from distutils.command.build import build
import os
from pathlib import Path

from pkg_resources import parse_requirements
from setuptools import find_namespace_packages
from setuptools import setup

required_packages = list(map(str, parse_requirements(Path("trulens_eval/requirements.txt").read_text())))
optional_packages = list(map(str, parse_requirements(Path("trulens_eval/requirements.optional.txt").read_text())))

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
      #  'build': javascript_build,
    },
    include_package_data=True, # includes things specified in MANIFEST.in
    packages=find_namespace_packages(
        include=["trulens_eval", "trulens_eval.*"]
    ),
    python_requires=
        '>= 3.8, < 3.12',  # Broken on python 3.12 release date for pyarrow. May be able to unpin if future deps are handled. make sure to also unpin conda python in ci-pr*.yaml
    entry_points={
        'console_scripts': [
            'trulens-eval=trulens_eval.utils.command_line:main'
        ],
    },
    install_requires=required_packages
)
