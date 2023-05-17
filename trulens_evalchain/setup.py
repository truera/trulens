from setuptools import find_namespace_packages
from setuptools import setup

setup(
    name="trulens_evalchain_test",
    packages=find_namespace_packages(include=["trulens_evalchain", "trulens_evalchain.*"]),
    python_requires='>=3.8',
    install_requires=[
            'cohere>=4.4.1',
            'datasets>=2.12.0',
            'python-dotenv>=1.0.0',
            'kaggle>=1.5.13',
            'langchain>=0.0.170',
            'merkle-json>=1.0.0',
            'openai>=0.27.6',
            'pinecone-client>=2.2.1',
            'pydantic>=1.10.7',
            'requests>=2.30.0',
            'slack-bolt>=1.18.0',
            'slack-sdk>=3.21.3',
            'streamlit>=1.22.0',
            'streamlit-aggrid>=0.3.4.post3',
            'streamlit-extras>=0.2.7',
            'tinydb>=4.7.1',
            'transformers>=4.10.0',
    ],
)
