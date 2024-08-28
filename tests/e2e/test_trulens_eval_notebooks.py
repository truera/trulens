"""Backwards compatibility test using notebooks from the pre-namespace package
release of trulens-eval."""

import os
from pathlib import Path
from subprocess import check_output
import sys
from unittest import TestCase

from dotenv import load_dotenv
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
from trulens.core.utils import imports as import_utils

LEGACY_NOTEBOOKS_PATH = Path(
    "_trulens_eval/trulens_eval/tests/docs_notebooks/notebooks_to_test"
).resolve()

if not LEGACY_NOTEBOOKS_PATH.exists():
    raise FileNotFoundError(
        f"Could not find legacy notebooks path: {LEGACY_NOTEBOOKS_PATH}. "
        "You may need to run `make _trulens_eval` to retrieve the old notebooks."
    )


def get_test_notebooks(path: Path):
    for file in path.iterdir():
        if file.suffix == ".ipynb":
            yield file


class KeysPreprocessor(ExecutePreprocessor):
    def __init__(self, timeout: int, kernel_name: str, env: dict):
        super().__init__(timeout=timeout, kernel_name=kernel_name)
        self.env = env

    def preprocess_cell(self, cell, resources, index, **kwargs):
        print(f"  Executing cell {index}.")
        if '"sk-..."' in cell["source"]:
            cell["source"] = cell["source"].replace(
                '"sk-..."', f"\"{self.env['OPENAI_API']}\""
            )
            print("    Replaced OPENAI_API key.")

        ret = super().preprocess_cell(cell, resources, index, **kwargs)

        return ret


class TestTruLensEvalNotebooks(TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv()

        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
        if OPENAI_API_KEY is None or not OPENAI_API_KEY.startswith("sk-"):
            raise ValueError(
                "OPENAI_API_KEY not found or not set correctly. Please set it in your .env before running this test."
            )
        HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", None)
        if HUGGINGFACE_API_KEY is None or not HUGGINGFACE_API_KEY.startswith(
            "hf_"
        ):
            raise ValueError(
                "HUGGINGFACE_API_KEY not found in environment. Please set it in your .env before running this test."
            )

        cls.env = {
            "OPENAI_API": OPENAI_API_KEY,
            "HUGGINGFACE_API": HUGGINGFACE_API_KEY,
        }

        # First make sure the python3 kernel is installed as we will be using it for each test.
        out = check_output(["jupyter", "kernelspec", "list"])
        if "python3" not in out.decode():
            raise ValueError(
                "Python3 kernel not found. Please install it before running this test."
            )

        # Check to make sure all trulens packages are installed.
        for package in [
            "trulens-core",
            "trulens-feedback",
            "trulens-dashboard",
            "trulens_eval",
            "trulens-benchmark",
            "trulens-instrument-langchain",
            "trulens-instrument-llamaindex",
            "trulens-instrument-nemo",
            "trulens-providers-bedrock",
            "trulens-providers-langchain",
            "trulens-providers-cortex",
            "trulens-providers-huggingface",
            "trulens-providers-openai",
            "trulens-providers-litellm",
        ]:
            cls.assertTrue(
                import_utils.is_package_installed(package),
                f"{package} is not installed.",
            )

    def _test_notebook(self, notebook_path: Path, kernel: str):
        with self.subTest(
            python=sys.version_info[0:2], notebook=notebook_path.name
        ):
            print(f"Legacy notebook: {notebook_path}")

            ep = KeysPreprocessor(timeout=600, kernel_name=kernel, env=self.env)

            with notebook_path.open() as f:
                nb = nbformat.read(f, as_version=4)
                try:
                    ep.preprocess(nb, {})
                except Exception as e:
                    self.fail(e)

    def test_legacy_notebooks(self):
        for notebook_path in get_test_notebooks(LEGACY_NOTEBOOKS_PATH):
            self._test_notebook(notebook_path, kernel="python3")
            self.fail("break for now")
