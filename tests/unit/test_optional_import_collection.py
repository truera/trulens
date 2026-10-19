import os
from pathlib import Path
import subprocess
import sys


def test_optional_dependency_skip_does_not_hide_consumer_modules() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "tests/unit/test_feedback.py",
            "tests/unit/test_otel_feedback_computation.py",
        ],
        cwd=repo_root,
        env={**os.environ, "TEST_OPTIONAL": "true"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "tests/unit/test_feedback.py::" in result.stdout
    assert "tests/unit/test_otel_feedback_computation.py::" in result.stdout
