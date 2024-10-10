"""Backwards compatibility test using notebooks from the pre-namespace package
release of trulens-eval.

These test need the old trulens_eval notebooks to be ready, checked out from an
old commit in workspace rooted at "_trulens_eval". See the Makefile target
called "_trulens_eval" on how this is done or just run that target.

The tests also require optional test dependencies to be installed. This can be
done with poetry:

```shell
poetry install --with tests-optional
```
"""

import os
from pathlib import Path
from subprocess import check_output
from typing import Dict
from unittest import TestCase
import warnings

from dotenv import load_dotenv
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat

from tests.test import optional_test

LEGACY_NOTEBOOKS_PATH = Path(
    "_trulens_eval/trulens_eval/tests/docs_notebooks/notebooks_to_test"
).resolve()


def get_test_notebooks(path: Path):
    for file in path.iterdir():
        if file.suffix == ".ipynb":
            yield file


class PatchesPreprocessor(ExecutePreprocessor):
    """Execute a notebook but make patches to the source before doing so.

    Args:
        timeout: The timeout for each cell execution.

        kernel_name: The name of the kernel to use for execution.

        patches: A dictionary of patches to make to the source code before
            executing the notebook. The keys are the strings to search for and
            the values are the strings to replace them with.

        eval_path: The path used when evaluating notebooks. If a notebook
            expects something in their local folder, make sure the path is set
            correctly.
    """

    def __init__(
        self, timeout: int, kernel_name: str, patches: dict, eval_path: Path
    ):
        super().__init__(
            timeout=timeout,
            kernel_name=kernel_name,
        )
        self.eval_path = eval_path
        self.patches = patches

    def preprocess(self, nb, resources, *args, **kwargs):
        if "metadata" not in resources:
            resources["metadata"] = {}
        resources["metadata"]["path"] = str(self.eval_path)
        return super().preprocess(nb, resources, *args, **kwargs)

    def preprocess_cell(self, cell, resources, index, **kwargs):
        first_line = cell["source"].split("\n")[0]

        print(f"  Executing cell {index}: {first_line}.")

        for patch_from, patch_to in self.patches.items():
            if patch_from in cell["source"]:
                cell["source"] = cell["source"].replace(patch_from, patch_to)
                print(f"    Patched: {patch_from}")

        try:
            ret = super().preprocess_cell(cell, resources, index, **kwargs)
        except Exception as e:
            print(f"    Failed: {e}")
            raise e

        return ret


class TestTruLensEvalNotebooks(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patches = None

        if not LEGACY_NOTEBOOKS_PATH.exists():
            warnings.warn(
                f"Could not find legacy notebooks path: {LEGACY_NOTEBOOKS_PATH}. "
                "No tests will run."
                "You may need to run `make _trulens_eval` to retrieve the old notebooks."
            )
            return

        load_dotenv()

        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
        if OPENAI_API_KEY is None or not OPENAI_API_KEY.startswith("sk-"):
            raise ValueError(
                "OPENAI_API_KEY not found or not set correctly. "
                "Please set it in your .env before running this test."
            )
        HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", None)
        if HUGGINGFACE_API_KEY is None or not HUGGINGFACE_API_KEY.startswith(
            "hf_"
        ):
            raise ValueError(
                "HUGGINGFACE_API_KEY not found in environment. "
                "Please set it in your .env before running this test."
            )

        # Some changes that need to be made to old notebooks to run independent of trulens_eval.
        cls.patches: Dict[str, str] = {
            'nltk.download("averaged_perceptron_tagger")': 'nltk.download("averaged_perceptron_tagger_eng")',
            "sk-...": OPENAI_API_KEY,
            'os.environ["OPENAI_API_KEY"] = "..."': f'os.environ["OPENAI_API_KEY"] = "{OPENAI_API_KEY}"',
        }

        # First make sure the python3 kernel is installed as we will be using it for each test.
        out = check_output(["jupyter", "kernelspec", "list"])
        if "python3" not in out.decode():
            raise ValueError(
                "Python3 kernel not found. Please install it before running this test."
            )

        cls.notebooks = {
            path.name: path
            for path in get_test_notebooks(LEGACY_NOTEBOOKS_PATH)
        }

    def _test_notebook(self, notebook_name: str, kernel: str = "python3"):
        if self.patches is None:
            self.skipTest(
                "Environment not configured for running old notebooks. "
                "Run `make _trulens_eval` first."
            )

        full_path = LEGACY_NOTEBOOKS_PATH / notebook_name

        if not full_path.exists():
            self.fail(f"Notebook {full_path} does not exist.")

        print(f"Legacy notebook: {notebook_name}")

        ep = PatchesPreprocessor(
            timeout=600,
            kernel_name=kernel,
            patches=self.patches,
            eval_path=LEGACY_NOTEBOOKS_PATH,
        )

        with full_path.open() as f:
            nb = nbformat.read(f, as_version=4)
            try:
                ep.preprocess(nb, {})
            except Exception as e:
                self.fail(e)

    # @skip("temp")
    def test_groundtruth_evals(self):
        self._test_notebook("groundtruth_evals.ipynb")

    # @skip("temp")
    def test_human_feedback(self):
        self._test_notebook("human_feedback.ipynb")

    # @skip("temp")
    @optional_test
    def test_langchain_faiss_example(self):
        self._test_notebook("langchain_faiss_example.ipynb")

    # @skip("temp")
    @optional_test
    def test_langchain_instrumentation(self):
        self._test_notebook("langchain_instrumentation.ipynb")

    # @skip("temp")
    @optional_test
    def test_langchain_quickstart(self):
        self._test_notebook("langchain_quickstart.ipynb")

    # @skip("temp")
    @optional_test
    def test_llama_index_instrumentation(self):
        self._test_notebook("llama_index_instrumentation.ipynb")

    # @skip("temp")
    @optional_test
    def test_llama_index_quickstart(self):
        self._test_notebook("llama_index_quickstart.ipynb")

    # @skip("temp")
    def test_logging(self):
        self._test_notebook("logging.ipynb")

    # @skip("temp")
    def test_prototype_evals(self):
        self._test_notebook("prototype_evals.ipynb")

    # @skip("temp")
    def test_quickstart(self):
        self._test_notebook("quickstart.ipynb")

    # @skip("temp")
    def test_text2text_quickstart(self):
        self._test_notebook("text2text_quickstart.ipynb")

    # @skip("temp")
    def test_trulens_instrumentation(self):
        self._test_notebook("trulens_instrumentation.ipynb")
