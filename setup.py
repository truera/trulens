import sys

from setuptools import find_namespace_packages
from setuptools import setup

version = "0.0.11"
versionArgument = "--customVersion"
if versionArgument in sys.argv:
    versionArgumentLoc = sys.argv.index(versionArgument)
    version = sys.argv[versionArgumentLoc + 1]
    #remove the flag and value
    sys.argv.pop(versionArgumentLoc)
    sys.argv.pop(versionArgumentLoc)

print("Generating package with version: " + version)

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="trulens",
    version=version,
    author="Truera Inc",
    author_email="all@truera.com",
    description=
    "Library containing attribution and interpretation methods for deep nets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="AGPL-3.0",
    url="https://truera.github.io/trulens/",
    packages=find_namespace_packages(include=["trulens", "trulens.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU Affero General Public License v3",
    ],
    python_requires='>=3.6'
)
