import glob
from os import listdir
import shutil
from typing import Iterable, Optional
from unittest import main

from dotenv import dotenv_values
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import NotebookNode
from nbformat import read
import pytest
from trulens.core.database.legacy import migration as legacy_migration


class VariableSettingPreprocessor(ExecutePreprocessor):
    def __init__(
        self,
        timeout: int,
        kernel_name: str,
        init_code: Optional[Iterable[str]] = None,
        code_to_run_before_each_cell: Optional[Iterable[str]] = None,
    ):
        super().__init__(timeout=timeout, kernel_name=kernel_name)

        if init_code:
            self.init_code = "\n".join(init_code) + "\n"
        else:
            self.init_code = ""
        self.ran_init_code = False

        if code_to_run_before_each_cell:
            self.code_to_run_before_each_cell = (
                "\n".join(code_to_run_before_each_cell) + "\n"
            )
        else:
            self.code_to_run_before_each_cell = ""

    def preprocess_cell(
        self, cell: NotebookNode, resources: dict, index: int, **kwargs
    ):
        if cell["cell_type"] == "code":
            if not self.ran_init_code:
                cell["source"] = self.init_code + cell["source"]
                self.ran_init_code = True
            cell["source"] = self.code_to_run_before_each_cell + cell["source"]
        ret = super().preprocess_cell(cell, resources, index, **kwargs)
        return ret


class DBMigrationPreprocessor(VariableSettingPreprocessor):
    def __init__(
        self,
        timeout: int,
        kernel_name: str,
        db_compat_version: str,
        init_code: Optional[Iterable[str]] = None,
        code_to_run_before_each_cell: Optional[Iterable[str]] = None,
    ):
        super().__init__(
            timeout=timeout,
            kernel_name=kernel_name,
            init_code=init_code,
            code_to_run_before_each_cell=code_to_run_before_each_cell,
        )
        shutil.copyfile(
            f"./release_dbs/{db_compat_version}/default.sqlite",
            "./default.sqlite",
        )

    def preprocess_cell(self, cell, resources, index, **kwargs):
        if "TruSession()" in cell["source"]:
            cell["source"] = (
                cell["source"]
                + "\nfrom trulens.core.session import TruSession\nsession=TruSession()\nsession.migrate_database()\n"
                + "\nfrom trulens.core.database.migrations.data import _sql_alchemy_serialization_asserts\n_sql_alchemy_serialization_asserts(session.connector.db)\n"
            )
        ret = super().preprocess_cell(cell, resources, index, **kwargs)

        return ret


NOTEBOOKS_TO_TEST = glob.glob(
    "./tests/docs_notebooks/notebooks_to_test/**/*.ipynb"
)


@pytest.mark.parametrize("filename", NOTEBOOKS_TO_TEST)
def test_notebook(filename):
    env: dict = dotenv_values("tests/docs_notebooks/.env")
    cell_start_code = ["import os"]
    for key, value in env.items():
        cell_start_code.append(f"os.environ['{key}'] = '{value}'")

    notebook_preprocessor = VariableSettingPreprocessor
    notebook_preprocessor_kwargs = {
        "timeout": 600,
        "kernel_name": "trulens-llm",
        "code_to_run_before_each_cell": cell_start_code,
    }
    with open(filename) as f:
        nb = read(f, as_version=4)
        notebook_preprocessor(**notebook_preprocessor_kwargs).preprocess(nb, {})


legacy_sqllite_migrations = [
    legacy_migration.migration_versions[0],
    legacy_migration.migration_versions[-1],
]

sqlalchemy_versions = [
    compat_versions
    for compat_versions in listdir("./release_dbs")
    if "sql_alchemy_" in compat_versions
]
migrations_to_test = legacy_sqllite_migrations + sqlalchemy_versions


@pytest.mark.parametrize("filename", NOTEBOOKS_TO_TEST)
@pytest.mark.parametrize("db_compat_version", migrations_to_test)
def test_notebook_backwards_compat(filename: str, db_compat_version: str):
    env: dict = dotenv_values("tests/docs_notebooks/.env")
    cell_start_code = ["import os"]
    for key, value in env.items():
        cell_start_code.append(f"os.environ['{key}'] = '{value}'")

    notebook_preprocessor = DBMigrationPreprocessor
    notebook_preprocessor_kwargs = {
        "timeout": 600,
        "kernel_name": "trulens-llm",
        "code_to_run_before_each_cell": cell_start_code,
        "db_compat_version": db_compat_version,
    }
    with open(filename) as f:
        nb = read(f, as_version=4)
        notebook_preprocessor(**notebook_preprocessor_kwargs).preprocess(nb, {})


if __name__ == "__main__":
    main()
