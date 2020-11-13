import datetime
from os import path
import sys
from setuptools import setup, find_namespace_packages

version = "0.0.4"
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
    name="netlens",
    version=version,
    author="Truera Inc",
    author_email="all@truera.com",
    description=
    "Library containing attribution and interpretation methods for deep nets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=["netlens", "netlens.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires='>=3.6')
