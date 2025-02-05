import os
from typing import Optional
import unittest

from nbconvert.preprocessors import ExecutePreprocessor
import nbformat


class TestOtelNotebooks(unittest.TestCase):
    @staticmethod
    def _run_and_test_notebook(
        notebook_path: str,
        timeout_in_seconds: int,
        kernel_name: Optional[str] = None,
    ) -> None:
        # Load notebook.
        with open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../../",
                notebook_path,
            )
        ) as f:
            nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)
        # Execute.
        if kernel_name:
            ep = ExecutePreprocessor(
                timeout=timeout_in_seconds, kernel_name=kernel_name
            )
        else:
            ep = ExecutePreprocessor(timeout=timeout_in_seconds)
        ep.preprocess(nb)

    def test_otel_exporter(self) -> None:
        # TODO(otel): This notebook should be made more expositional and made
        #             into a doc.
        self._run_and_test_notebook(
            os.path.join("examples/experimental/otel_exporter.ipynb"),
            timeout_in_seconds=300,
        )


if __name__ == "__main__":
    unittest.main()
