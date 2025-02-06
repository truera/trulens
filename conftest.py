import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run_basic_tests",
        action="store_true",
        default=True,
        help="Run tests not marked optional/snowflake",
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


def pytest_collection_modifyitems(config, items):
    basic = config.getoption("--run_basic_tests") or os.environ.get(
        "TEST_BASIC", ""
    ).lower() in ["1", "true"]
    optional = config.getoption("--run_optional_tests") or os.environ.get(
        "TEST_OPTIONAL", ""
    ).lower() in ["1", "true"]
    snowflake = config.getoption("--run_snowflake_tests") or os.environ.get(
        "TEST_SNOWFLAKE", ""
    ).lower() in ["1", "true"]

    skip_basic = pytest.mark.skip(
        reason="Skipping non optional/snowflake tests"
    )
    skip_optional = pytest.mark.skip(reason="Skipping optional tests")
    skip_snowflake = pytest.mark.skip(reason="Skipping snowflake tests")

    for item in items:
        # Assume that `item` is marked with at most one of
        # required_only/optional/snowflake.
        if (
            len([
                curr
                for curr in ["required_only", "optional", "snowflake"]
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
        elif "optional" in item.keywords():
            if not optional:
                item.add_marker(skip_optional)
        elif "snowflake" in item.keywords():
            if not snowflake:
                item.add_marker(skip_snowflake)
        else:
            if not basic:
                item.add_marker(skip_basic)
