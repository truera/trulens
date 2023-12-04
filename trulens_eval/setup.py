from distutils import log
from distutils.command.build import build
import os

from setuptools import find_namespace_packages
from setuptools import setup


class javascript_build(build):

    def run(self):
        log.info("running npm i")
        os.system("npm i --prefix trulens_eval/react_components/record_viewer")
        log.info("running npm run build")
        os.system(
            "npm run --prefix trulens_eval/react_components/record_viewer build"
        )
        build.run(self)


langchain_version = "0.0.335"  # duplicated in trulens_eval.utils.imports, don't know how to dedup
llama_version = "0.8.69"  # duplicated in trulens_eval.utils.imports, don't know how to dedup

setup(
    name="trulens_eval",
    cmdclass={
        'build': javascript_build,
    },
    include_package_data=True,
    packages=find_namespace_packages(
        include=["trulens_eval", "trulens_eval.*"]
    ),
    python_requires=
    '>=3.8,<3.12',  # Broken on python 3.12 release date for pyarrow. May be able to unpin if future deps are handled. make sure to also unpin conda python in ci-pr*.yaml
    entry_points={
        'console_scripts': [
            'trulens-eval=trulens_eval.utils.command_line:main'
        ],
    },
    install_requires=[
        'cohere>=4.4.1',
        'datasets>=2.12.0',
        'python-dotenv>=1.0.0',
        'kaggle>=1.5.13',
        f'langchain>={langchain_version}',  # required for cost tracking even outside of langchain
        'typing-inspect>=0.8.0',  # langchain with python < 3.9 fix
        'typing_extensions>=4.5.0',  # langchain with python < 3.9 fix
        f'llama_index>={llama_version}',
        # VectorStoreIndex changes need to be taken account for in later versions
        'html2text>=2020.1.16',  # for llama_index example using SimpleWebPageReader
        'merkle-json>=1.0.0',
        'millify>=0.1.1',
        'openai >=1.1.1, <2',
        # NOTE(piotrm): openai v2 likely to break things
        'httpx >= 0.23.3',  # for openai client serialization
        'pinecone-client>=2.2.1',
        'pydantic >=1.10.7, <2',
        # TODO(piotrm): need some migration for pydantic 2
        'humanize>=4.6.0',
        'slack-bolt>=1.18.0',
        'slack-sdk>=3.21.3',
        'streamlit>=1.27.0',
        'streamlit-aggrid>=0.3.4.post3',
        'streamlit-extras>=0.2.7',
        'streamlit-javascript>=0.1.5',  # for copy to clipboard functionality (in progress)
        'transformers>=4.10.0',
        'frozendict>=2.3.8',
        'munch>=3.0.0',
        'ipywidgets>=8.0.6',
        'numpy>=1.23.5',
        'sqlalchemy>=2.0.19',
        'alembic>=1.11.2',

        # 'nest_asyncio>=1.5.6',
        # NOTE(piotrm): disabling for now, need more investigation of compatibility issues
    ],
)
