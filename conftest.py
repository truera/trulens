import os

import pytest

# Initialize langchain globals for langchain 1.x compatibility
# This must happen before any langchain code is imported
try:
    from langchain_core import globals as langchain_globals

    langchain_globals.set_debug(False)
    langchain_globals.set_verbose(False)
except (ImportError, AttributeError):
    # Fallback if langchain not installed or API changed
    pass


def pytest_addoption(parser):
    """Register custom command-line options for the TruLens test suite.

    Adds the following flags:

    - ``--skip_basic_tests``: Skip tests not marked optional or snowflake.
    - ``--run_optional_tests``: Enable tests marked as optional.
    - ``--run_snowflake_tests``: Enable tests marked as snowflake.
    - ``--run_huggingface_tests``: Enable tests marked as huggingface.

    Each flag has a corresponding environment-variable override
    (``SKIP_BASIC_TESTS``, ``TEST_OPTIONAL``, ``TEST_SNOWFLAKE``,
    ``TEST_HUGGINGFACE``) that is honoured by
    :func:`pytest_collection_modifyitems`.
    """
    parser.addoption(
        "--skip_basic_tests",
        action="store_true",
        default=False,
        help="Skip tests not marked optional/snowflake",
    )
    parser.addoption(
        "--run_optional_tests",
        action="store_true",
        default=False,
        help="Run tests marked as optional",
    )
    parser.addoption(
        "--run_snowflake_tests",
        action="store_true",
        default=False,
        help="Run tests marked as snowflake",
    )
    parser.addoption(
        "--run_huggingface_tests",
        action="store_true",
        default=False,
        help="Run tests marked as huggingface",
    )


def pytest_collection_modifyitems(config, items):
    """Filter collected test items based on CLI flags and environment variables.

    Applies skip markers to tests according to the following rules:

    - Tests marked ``required_only`` are skipped when optional or snowflake
      test modes are active, or when basic tests are disabled.
    - Tests marked ``optional`` are skipped unless ``--run_optional_tests`` is
      set or the ``TEST_OPTIONAL`` environment variable is truthy.
    - Tests marked ``snowflake`` are skipped unless ``--run_snowflake_tests``
      is set or the ``TEST_SNOWFLAKE`` environment variable is truthy.
    - Tests marked ``huggingface`` are skipped unless
      ``--run_huggingface_tests`` is set or the ``TEST_HUGGINGFACE``
      environment variable is truthy.
    - Unmarked tests are skipped when basic tests are disabled via
      ``--skip_basic_tests`` or the ``SKIP_BASIC_TESTS`` environment variable.

    Args:
        config: The pytest configuration object providing access to CLI options.
        items: The list of collected test ``Item`` objects to be modified
            in-place.

    Raises:
        ValueError: If a test item is marked with more than one of
            ``required_only``, ``optional``, ``snowflake``, or ``huggingface``.
    """
    basic = not config.getoption("--skip_basic_tests") and os.environ.get(
        "SKIP_BASIC_TESTS", ""
    ).lower() not in ["1", "true"]
    optional = config.getoption("--run_optional_tests") or os.environ.get(
        "TEST_OPTIONAL", ""
    ).lower() in ["1", "true"]
    snowflake = config.getoption("--run_snowflake_tests") or os.environ.get(
        "TEST_SNOWFLAKE", ""
    ).lower() in ["1", "true"]
    huggingface = config.getoption("--run_huggingface_tests") or os.environ.get(
        "TEST_HUGGINGFACE", ""
    ).lower() in ["1", "true"]

    skip_basic = pytest.mark.skip(
        reason="Skipping non optional/snowflake tests"
    )
    skip_optional = pytest.mark.skip(reason="Skipping optional tests")
    skip_snowflake = pytest.mark.skip(reason="Skipping snowflake tests")
    skip_huggingface = pytest.mark.skip(reason="Skipping huggingface tests")

    for item in items:
        # Assume that `item` is marked with at most one of
        # required_only/optional/snowflake.
        if (
            len([
                curr
                for curr in [
                    "required_only",
                    "optional",
                    "snowflake",
                    "huggingface",
                ]
                if curr in item.keywords
            ])
            > 1
        ):
            raise ValueError(
                "Test marked with multiple of required_only/optional/snowflake!"
            )
        if "required_only" in item.keywords:
            if optional or snowflake:
                item.add_marker(
                    pytest.mark.skip(
                        reason="Skipping as optional/snowflake tests are running"
                    )
                )
            if not basic:
                item.add_marker(skip_basic)
        elif "optional" in item.keywords:
            if not optional:
                item.add_marker(skip_optional)
        elif "snowflake" in item.keywords:
            if not snowflake:
                item.add_marker(skip_snowflake)
        elif "huggingface" in item.keywords:
            if not huggingface:
                item.add_marker(skip_huggingface)
        else:
            if not basic:
                item.add_marker(skip_basic)
