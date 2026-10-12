"""Leaderboard aggregates must not require the optional feedback package.

`trulens.core` installs without `trulens.feedback`, so anything in core that
reads the unparsable-score sentinel has to take the value from
`trulens.core.utils.constants` rather than `trulens.feedback.llm_provider`.
Getting this wrong raises `ModuleNotFoundError` from
`get_leaderboard_aggregates()`, which is reachable on a core-only install with
no app and no feedback recorded.

The subprocess test is the real guard: a source-text scan cannot tell whether a
reference is executed, and only an environment where `trulens.feedback` is
genuinely unimportable reproduces the failure.
"""

import ast
import subprocess
import sys

import pytest


def test_leaderboard_module_does_not_import_optional_feedback():
    """No executed reference to `trulens.feedback` may remain in core."""
    pytest.importorskip("trulens.core")

    from trulens.core.database import sqlalchemy as db_sqlalchemy

    with open(db_sqlalchemy.__file__) as f:
        source = f.read()

    # Parsed rather than grepped: a docstring may legitimately mention
    # trulens.feedback, an executed import may not. AST keeps those apart
    # without depending on indentation or backticks.
    tree = ast.parse(source)
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [
                f"import {a.name}"
                for a in node.names
                if "trulens.feedback" in a.name
            ]
        elif isinstance(node, ast.ImportFrom):
            if node.module and "trulens.feedback" in node.module:
                offenders.append(f"from {node.module} import ...")
    assert not offenders, (
        "trulens.core.database.sqlalchemy still reaches into the optional "
        f"trulens.feedback package: {offenders}"
    )


def test_core_only_install_can_read_an_empty_leaderboard():
    """With `trulens.feedback` unimportable, an empty leaderboard must not raise."""
    pytest.importorskip("trulens.core")

    script = """
import sys

class Blocker:
    def find_module(self, name, path=None):
        return self if name == "trulens.feedback" else None

    def load_module(self, name):
        raise ImportError("trulens.feedback is blocked for this test")

sys.meta_path.insert(0, Blocker())

import os
import tempfile

from trulens.core.database import sqlalchemy as db_sqlalchemy

d = tempfile.mkdtemp()
db = db_sqlalchemy.SQLAlchemyDB.from_db_url(
    "sqlite:///" + os.path.join(d, "trulens.sqlite")
)
db.migrate_database()
df, cols = db.get_leaderboard_aggregates()
assert df.empty, df.shape
assert cols == [], cols
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert "OK" in result.stdout, (
        f"core-only leaderboard read failed:\n{result.stdout}\n{result.stderr}"
    )
