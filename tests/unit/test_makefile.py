"""Regression tests for the unit-test command's exit status."""

import os
import pathlib
import shutil
import subprocess
import textwrap

import pytest


def _run_unit_tests(
    tmp_path: pathlib.Path,
    *,
    has_xdist: bool,
    otel_exit: int,
    other_exit: int,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run the real Makefile with controlled Poetry command outcomes."""
    make = shutil.which("gmake") or shutil.which("make")
    if make is None or not pathlib.Path("/bin/bash").exists():
        pytest.skip("The repository Makefile requires GNU Make and Bash.")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    poetry = bin_dir / "poetry"
    poetry.write_text(
        textwrap.dedent(
            """\
            #!/bin/sh
            if [ "$1" = run ] && [ "$2" = python ]; then
                exit "$TRULENS_TEST_XDIST_EXIT"
            fi
            if [ "$1" != run ] || [ "$2" != pytest ]; then
                exit 99
            fi
            shift 2
            phase=other
            for arg in "$@"; do
                case "$arg" in
                    tests/unit/test_otel*.py) phase=otel ;;
                esac
            done
            printf '%s\\n' "$phase" >> "$TRULENS_TEST_PHASE_LOG"
            if [ "$phase" = otel ]; then
                exit "$TRULENS_TEST_OTEL_EXIT"
            fi
            exit "$TRULENS_TEST_OTHER_EXIT"
            """
        ),
        encoding="utf-8",
    )
    poetry.chmod(0o755)
    phase_log = tmp_path / "phases.log"
    phase_log.touch()

    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "TRULENS_TEST_XDIST_EXIT": "0" if has_xdist else "1",
        "TRULENS_TEST_OTEL_EXIT": str(otel_exit),
        "TRULENS_TEST_OTHER_EXIT": str(other_exit),
        "TRULENS_TEST_PHASE_LOG": str(phase_log),
    }
    # This may run inside make test-unit; isolate it from the outer make's
    # jobserver and flags such as --dry-run or --ignore-errors.
    for name in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "GNUMAKEFLAGS"):
        env.pop(name, None)

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [make, "--no-print-directory", "test-unit"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result, phase_log.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("has_xdist", [True, False])
def test_unit_stops_when_otel_tests_fail(tmp_path, has_xdist):
    """An OTEL failure must fail make and stop the remaining test phase."""
    result, phases = _run_unit_tests(
        tmp_path, has_xdist=has_xdist, otel_exit=7, other_exit=0
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert phases == ["otel"], output


@pytest.mark.parametrize("has_xdist", [True, False])
def test_unit_fails_when_remaining_tests_fail(tmp_path, has_xdist):
    """The second phase must still fail make after OTEL tests pass."""
    result, phases = _run_unit_tests(
        tmp_path, has_xdist=has_xdist, otel_exit=0, other_exit=9
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert phases == ["otel", "other"], output


@pytest.mark.parametrize("has_xdist", [True, False])
def test_unit_succeeds_when_both_test_phases_pass(tmp_path, has_xdist):
    """Both phases must run and succeed, including without pytest-xdist."""
    result, phases = _run_unit_tests(
        tmp_path, has_xdist=has_xdist, otel_exit=0, other_exit=0
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert phases == ["otel", "other"], output
