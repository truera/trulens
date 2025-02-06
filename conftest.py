import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--optional",
        action="store_true",
        default=False,
        help="Run tests marked as optional",
    )
    parser.addoption(
        "--snowflake",
        action="store_true",
        default=False,
        help="Run tests marked as snowflake",
    )
    parser.addoption(
        "--all", action="store_true", default=False, help="Run all tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--all"):
        # If --all is specified, do not filter any tests
        return

    optional = config.getoption("--optional") or os.environ.get(
        "TEST_OPTIONAL", ""
    ).lower() in ["1", "true"]
    snowflake = config.getoption("--snowflake") or os.environ.get(
        "TEST_SNOWFLAKE", ""
    ).lower() in ["1", "true"]

    skip_optional = pytest.mark.skip(reason="Skipping optional tests")
    skip_snowflake = pytest.mark.skip(reason="Skipping snowflake tests")

    for item in items:
        if optional and "optional" in item.keywords:
            continue
        elif snowflake and "snowflake" in item.keywords:
            continue
        elif not optional and "optional" in item.keywords:
            item.add_marker(skip_optional)
        elif not snowflake and "snowflake" in item.keywords:
            item.add_marker(skip_snowflake)
