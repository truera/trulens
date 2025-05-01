import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest import TestCase

import pandas as pd

_MEMORY_LOCATION_REGEX_REPLACEMENT = (
    r"<([0-9a-zA-Z\._-]+) object at 0x[0-9a-fA-F]+>",
    r"<\1 at memory_location>",
)


def compare_dfs_accounting_for_ids_and_timestamps(
    test_case: TestCase,
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    ignore_locators: Optional[Sequence[str]],
    timestamp_tol: pd.Timedelta = pd.Timedelta(0),
    ignore_memory_locations: bool = True,
    regex_replacements: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """
    Compare two Dataframes are equal, accounting for ids and timestamps. That
    is:
    1. The ids between the two Dataframes may be different, but they have to be
       consistent. That is, if one Dataframe reuses an id in two places, then
       the other must as well.
    2. The timestamps between the two Dataframes may be different, but they
       have to be in the same order.

    Args:
        test_case: unittest.TestCase instance to use for assertions
        expected: expected results
        actual: actual results
        ignore_locators: locators to ignore when comparing the Dataframes
        timestamp_tol: the tolerance for comparing timestamps
        ignore_memory_locations: whether to ignore memory locations in strings (e.g. "0x1234abcd")
        regex_replacements: a list of tuples of (regex, replacement) to apply to all strings
    """
    id_mapping: Dict[str, str] = {}
    timestamp_mapping: Dict[pd.Timestamp, pd.Timestamp] = {}
    test_case.assertEqual(len(expected), len(actual))
    test_case.assertListEqual(list(expected.columns), list(actual.columns))
    if not regex_replacements:
        regex_replacements = []
    if ignore_memory_locations:
        regex_replacements.append(_MEMORY_LOCATION_REGEX_REPLACEMENT)
    for i in range(len(expected)):
        for col in expected.columns:
            _compare_entity(
                test_case,
                expected.iloc[i][col],
                actual.iloc[i][col],
                id_mapping,
                timestamp_mapping,
                is_id=col.endswith("_id"),
                locator=f"df.iloc[{i}][{col}]",
                ignore_locators=ignore_locators,
                regex_replacements=regex_replacements,
            )
    # Ensure that the id mapping is a bijection.
    test_case.assertEqual(
        len(set(id_mapping.values())),
        len(id_mapping),
        "Ids are not a bijection!",
    )
    # Ensure that the timestamp mapping is strictly increasing.
    prev_value = None
    for curr in sorted(timestamp_mapping.keys()):
        if prev_value is not None:
            test_case.assertLess(
                prev_value - timestamp_tol,
                timestamp_mapping[curr],
                "Timestamps are not in the same order!",
            )
        prev_value = timestamp_mapping[curr]


def _jsonifiable(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def _compare_entity(
    test_case: TestCase,
    expected: Any,
    actual: Any,
    id_mapping: Dict[str, str],
    timestamp_mapping: Dict[pd.Timestamp, pd.Timestamp],
    is_id: bool,
    locator: str,
    ignore_locators: Optional[Sequence[str]],
    regex_replacements: List[Tuple[str, str]],
) -> None:
    if ignore_locators and locator in ignore_locators:
        return
    test_case.assertEqual(
        type(expected),
        type(actual),
        f"Types of {locator} do not match!\nEXPECTED: {type(expected)}\nACTUAL: {type(actual)}",
    )
    if is_id:
        test_case.assertEqual(
            type(expected), str, f"Type of id {locator} is not a string!"
        )
        if expected not in id_mapping:
            id_mapping[expected] = actual
        test_case.assertEqual(
            id_mapping[expected],
            actual,
            f"Ids of {locator} are not consistent!",
        )
    elif isinstance(expected, list):
        test_case.assertEqual(len(expected), len(actual))
        for i in range(len(expected)):
            _compare_entity(
                test_case,
                expected[i],
                actual[i],
                id_mapping,
                timestamp_mapping,
                is_id=is_id,
                locator=f"{locator}[{i}]",
                ignore_locators=ignore_locators,
                regex_replacements=regex_replacements,
            )
    elif isinstance(expected, dict):
        test_case.assertEqual(
            expected.keys(),
            actual.keys(),
            f"Keys of {locator} do not match!\nEXPECTED: {expected.keys()}\nACTUAL: {actual.keys()}",
        )
        for k in expected.keys():
            _compare_entity(
                test_case,
                expected[k],
                actual[k],
                id_mapping,
                timestamp_mapping,
                is_id=k.endswith("_id"),
                locator=f"{locator}[{k}]",
                ignore_locators=ignore_locators,
                regex_replacements=regex_replacements,
            )
    elif isinstance(expected, pd.Timestamp):
        if expected not in timestamp_mapping:
            timestamp_mapping[expected] = actual
        test_case.assertEqual(
            timestamp_mapping[expected],
            actual,
            f"Timestamps of {locator} are not consistent!",
        )
    else:
        if isinstance(expected, str):
            if _jsonifiable(expected):
                expected = json.loads(expected)
                actual = json.loads(actual)
                _compare_entity(
                    test_case,
                    expected,
                    actual,
                    id_mapping,
                    timestamp_mapping,
                    is_id=is_id,
                    locator=locator,
                    ignore_locators=ignore_locators,
                    regex_replacements=regex_replacements,
                )
                return
            # TODO(this_pr): do we still need this?
            for regex, replacement in regex_replacements:
                expected = re.sub(regex, replacement, expected)
                actual = re.sub(regex, replacement, actual)
        test_case.assertEqual(
            expected,
            actual,
            f"{locator} does not match!\nEXPECTED: {expected}\nACTUAL: {actual}",
        )
