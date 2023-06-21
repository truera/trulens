import os
from os import listdir
from typing import Sequence
from unittest import main
from unittest import TestCase

from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import read


class DocsNotebookTests(TestCase):
    pass


class VariableSettingPreprocessor(ExecutePreprocessor):

    def __init__(
        self, timeout: int, kernel_name: str,
        code_to_run_before_each_cell: Sequence[str]
    ):
        super().__init__(timeout=timeout, kernel_name=kernel_name)
        self.code_to_run_before_each_cell = "\n".join(
            code_to_run_before_each_cell
        ) + "\n"

    def preprocess_cell(self, cell, resources, index, **kwargs):
        if cell["cell_type"] == "code":
            cell["source"] = self.code_to_run_before_each_cell + cell["source"]
        ret = super().preprocess_cell(cell, resources, index, **kwargs)
        return ret


def get_unit_test_for_filename(filename):

    def test(self):
        with open(f'./tests/docs_notebooks/notebooks_to_test/{filename}') as f:
            OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
            HUGGINGFACE_API_KEY = os.environ['HUGGINGFACE_API_KEY']
            nb = read(f, as_version=4)
            VariableSettingPreprocessor(
                timeout=600,
                kernel_name='trulens-llm',
                code_to_run_before_each_cell=[
                    f"import os",
                    f"os.environ['OPENAI_API_KEY']='{OPENAI_API_KEY}'",
                    f"os.environ['HUGGINGFACE_API_KEY']='{HUGGINGFACE_API_KEY}'",
                ]
            ).preprocess(nb, {})

    return test


for filename in listdir('./tests/docs_notebooks/notebooks_to_test/'):
    if filename.endswith('.ipynb'):
        setattr(
            DocsNotebookTests, 'test_' + filename.split('.ipynb')[0],
            get_unit_test_for_filename(filename)
        )

# Test Backwards Compat (TODO: Move this to CI/CD)
import shutil
shutil.copyfile("./release_dbs/0.1.2/default.sqlite", "./default.sqlite")

if __name__ == '__main__':
    main()
