"""Backwards compatibility test using notebooks from the pre-namespace package
release of trulens-eval.

These test need the old trulens_eval notebooks to be ready, checked out from an
old commit in workspace rooted at "_trulens_eval". See the Makefile target
called "_trulens_eval" on how this is done or just run that target.

The tests also require optional test dependencies to be installed. This can be
done with poetry:

```shell poetry install --with tests-optional ```
"""

import os
from pathlib import Path
from subprocess import check_output
import sys
from typing import Dict
from unittest import TestCase
from unittest import skip

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


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
if OPENAI_API_KEY is None or not OPENAI_API_KEY.startswith("sk-"):
    raise ValueError(
        "OPENAI_API_KEY not found or not set correctly. Please set it in your .env before running this test."
    )
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", None)
if HUGGINGFACE_API_KEY is None or not HUGGINGFACE_API_KEY.startswith("hf_"):
    raise ValueError(
        "HUGGINGFACE_API_KEY not found in environment. Please set it in your .env before running this test."
    )

# Some changes that need to be made to old notebooks to run independent of trulens_eval.
PACHES: Dict[str, str] = {
    'nltk.download("averaged_perceptron_tagger")': 'nltk.download("averaged_perceptron_tagger_eng")',
    "sk-...": OPENAI_API_KEY,
    'os.environ["OPENAI_API_KEY"] = "..."': f'os.environ["OPENAI_API_KEY"] = "{OPENAI_API_KEY}"',
}


class KeysPreprocessor(ExecutePreprocessor):
    def __init__(
        self, timeout: int, kernel_name: str, env: dict, eval_path: Path
    ):
        super().__init__(
            timeout=timeout,
            kernel_name=kernel_name,
        )
        self.eval_path = eval_path
        self.env = env

    def preprocess(self, nb, resources, *args, **kwargs):
        if "metadata" not in resources:
            resources["metadata"] = {}
        resources["metadata"]["path"] = str(self.eval_path)
        return super().preprocess(nb, resources, *args, **kwargs)

    def preprocess_cell(self, cell, resources, index, **kwargs):
        first_line = cell["source"].split("\n")[0]

        print(f"  Executing cell {index}: {first_line}.")

        for patch_from, path_to in PACHES.items():
            if patch_from in cell["source"]:
                cell["source"] = cell["source"].replace(patch_from, path_to)
                print(f"    Patched: {patch_from}")

        ret = super().preprocess_cell(cell, resources, index, **kwargs)

        return ret


class TestTruLensEvalNotebooks(TestCase):
    @classmethod
    def setUpClass(cls):
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
        # TODO: Check that the optional packages (extras) are installed?
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

        cls.notebooks = {
            path.name: path
            for path in get_test_notebooks(LEGACY_NOTEBOOKS_PATH)
        }

    def _test_notebook(self, notebook_path: Path, kernel: str = "python3"):
        with self.subTest(
            python=sys.version_info[0:2], notebook=notebook_path.name
        ):
            print(f"Legacy notebook: {notebook_path}")

            ep = KeysPreprocessor(
                timeout=600,
                kernel_name=kernel,
                env=self.env,
                eval_path=LEGACY_NOTEBOOKS_PATH,
            )

            with notebook_path.open() as f:
                nb = nbformat.read(f, as_version=4)
                try:
                    ep.preprocess(nb, {})
                except Exception as e:
                    self.fail(e)

    # @skip("Not working yet")
    def test_groundtruth_evals(self):
        self._test_notebook(self.notebooks["groundtruth_evals.ipynb"])

    @skip("Done")
    def test_human_feedback(self):
        self._test_notebook(self.notebooks["human_feedback.ipynb"])

    @skip("Done")
    def test_langchain_faiss_example(self):
        self._test_notebook(self.notebooks["langchain_faiss_example.ipynb"])

    @skip("Done")
    def test_langchain_instrumentation(self):
        self._test_notebook(self.notebooks["langchain_instrumentation.ipynb"])

    @skip("Done")
    def test_langchain_quickstart(self):
        self._test_notebook(self.notebooks["langchain_quickstart.ipynb"])

    @skip("Done")
    def test_llama_index_instrumentation(self):
        self._test_notebook(self.notebooks["llama_index_instrumentation.ipynb"])

    @skip("Done")
    def test_llama_index_quickstart(self):
        self._test_notebook(self.notebooks["llama_index_quickstart.ipynb"])

    @skip("Done")
    def test_logging(self):
        self._test_notebook(self.notebooks["logging.ipynb"])

    @skip("Done")
    def test_prototype_evals(self):
        self._test_notebook(self.notebooks["prototype_evals.ipynb"])

    @skip("Done")
    def test_quickstart(self):
        self._test_notebook(self.notebooks["quickstart.ipynb"])

    @skip("Done")
    def test_text2text_quickstart(self):
        self._test_notebook(self.notebooks["text2text_quickstart.ipynb"])

    @skip("Done")
    def test_trulens_instrumentation(self):
        self._test_notebook(self.notebooks["trulens_instrumentation.ipynb"])

    # def test_legacy_notebooks(self):
    #    for notebook_path in get_test_notebooks(LEGACY_NOTEBOOKS_PATH):
    #        self._test_notebook(notebook_path, kernel="python3")
    #        self.fail("break for now")
