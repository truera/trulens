from setuptools import find_namespace_packages
from setuptools import setup

setup(
    name='trulens',
    packages=find_namespace_packages(include=['trulens', 'trulens.*']),
    python_requires='>=3.8',
    install_requires=['numpy>=1.23.5']
)
