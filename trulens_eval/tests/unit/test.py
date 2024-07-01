from dataclasses import fields
from dataclasses import is_dataclass
from datetime import datetime
import functools
import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Set
import unittest
from unittest import TestCase

import pydantic
from pydantic import BaseModel
import yaml

from trulens_eval.utils.python import caller_frame
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.serial import JSON_BASES
from trulens_eval.utils.serial import Lens

OPTIONAL_ENV_VAR = "TEST_OPTIONAL"
"""Env var that were to evaluate to true indicates that optional tests are to be
run."""

WRITE_GOLDEN_VAR = "WRITE_GOLDEN"
"""Env var for indicating whether golden expected results are to be written (if
true) or read and compared (if false/undefined)."""


def optional_test(testmethodorclass):
    """Only run the decorated test if the environment variable with_optional
    evalutes true.
    
    These are meant to be run only in an environment where optional packages
    have been installed.
    """

    return unittest.skipIf(
        not os.environ.get(OPTIONAL_ENV_VAR), "optional test"
    )(testmethodorclass)


def requiredonly_test(testmethodorclass):
    """
    Only runs the decorated test if the environment variable with_optional
    evalutes to false or is not set.
    
    Decorated tests are meant to run specifically when optional imports are not
    installed.
    """

    return unittest.skipIf(
        os.environ.get(OPTIONAL_ENV_VAR), "not an optional test"
    )(testmethodorclass)


def module_installed(module: str) -> bool:
    """Check if a module is installed."""

    try:
        __import__(module)
        return True
    except ImportError:
        return False


class JSONTestCase(TestCase):
    """TestCase class that adds JSON comparisons and golden expectation handling."""

    def assertGoldenJSONEqual(
        self,
        actual: JSON,
        golden_filename: str,
        skips: Optional[Set[str]] = None,
        numeric_places: int = 7,
    ):
        """Assert equality between [JSON-like][trulens_eval.util.serial.JSON]
        `actual` and the content of `golden_filename`.

        If the environment variable
        [WRITE_GOLDEN_VAR][trulens_eval.tests.unit.test.WRITE_GOLDEN_VAR] is
        set, the golden file will be overwritten with the `actual` content. See
        [assertJSONEqual][trulens_eval.tests.unit.test.assertJSONEqual] for
        details on the equality check.

        Args:
            actual: The actual JSON-like object produced by some test.

            golden_filename: The name of the golden file to compare against that
            stores the expected JSON-like results for the test.

            skips: A set of keys to skip in the comparison.

            numeric_places: The number of decimal places to compare for floating
                point

        Raises:
            FileNotFoundError: If the golden file is not found.

            AssertionError: If the actual JSON-like object does not match the
                expected JSON-like object

            AssertionError: If the golden file is written.  
        """

        write_golden: bool = bool(os.environ.get(WRITE_GOLDEN_VAR, ""))

        caller_path = Path(caller_frame(offset=1).f_code.co_filename).parent
        golden_path = (caller_path / "golden" / golden_filename).resolve()

        if golden_path.suffix == ".json":
            writer = json.dump
            loader = json.load
        elif golden_path.suffix == ".yaml":
            writer = yaml.dump
            loader = functools.partial(yaml.load, Loader=yaml.FullLoader)
        else:
            raise ValueError(
                f"Unknown file extension {golden_path.suffix}."
            )

        if write_golden:
            with golden_path.open("w") as f:
                writer(actual, f)

            self.fail("Golden file written.")

        else:
            if not golden_path.exists():
                raise FileNotFoundError(f"Golden file {golden_path} not found.")

            with golden_path.open("r") as f:
                expected = loader(f)

            self.assertJSONEqual(
                actual, expected, skips=skips, numeric_places=numeric_places
            )

    def assertJSONEqual(
        self,
        j1: JSON,
        j2: JSON,
        path: Optional[Lens] = None,
        skips: Optional[Set[str]] = None,
        numeric_places: int = 7
    ) -> None:
        """Assert equality between JSON-like `j1` and `j2`.
        
        The `path` argument is used to track the path to the current object in
        the JSON structure. It is used to provide more informative error
        messages in case of a mismatch. The `skips` argument is used to skip
        certain keys in the comparison. The `numeric_places` argument is used to
        specify the number of decimal places to compare for floating point
        numbers.

        Data types supported for comparison are:
            - JSON-like base types (int, float, str)
            - JSON-like constructors (list, dict)
            - datetime
            - dataclasses
            - pydantic models
        
        Args:
            j1: The first JSON-like object.

            j2: The second JSON-like object.
            
            path: The path to the current object in the JSON structure.

            skips: A set of keys to skip in the comparison.

            numeric_places: The number of decimal places to compare for floating
                point numbers.

        Raises:
            AssertionError: If the two JSON-like objects are
                not equal (except for anything skipped) or anything within
                numeric tolerance.
        """

        skips = skips or set([])
        path = path or Lens()

        def recur(j1, j2, path):
            return self.assertJSONEqual(
                j1, j2, path=path, skips=skips, numeric_places=numeric_places
            )

        ps = str(path)

        self.assertIsInstance(j1, type(j2), ps)

        if isinstance(j1, JSON_BASES):
            if isinstance(j1, (int, float)):
                self.assertAlmostEqual(j1, j2, places=numeric_places, msg=ps)
            else:
                self.assertEqual(j1, j2, ps)

        elif isinstance(j1, Dict):

            ks1 = set(j1.keys())
            ks2 = set(j2.keys())

            self.assertSetEqual(ks1, ks2, ps)

            for k in ks1:
                if k in skips:
                    continue

                recur(j1[k], j2[k], path=path[k])

        elif isinstance(j1, Sequence):
            self.assertEqual(len(j1), len(j2), ps)

            for i, (v1, v2) in enumerate(zip(j1, j2)):
                recur(v1, v2, path=path[i])

        elif isinstance(j1, datetime):
            self.assertEqual(j1, j2, ps)

        elif is_dataclass(j1):
            for f in fields(j1):
                if f.name in skips:
                    continue

                self.assertTrue(hasattr(j2, f.name))

                recur(getattr(j1, f.name), getattr(j2, f.name), path[f.name])

        elif isinstance(j1, BaseModel):
            for f in j1.model_fields:
                if f in skips:
                    continue

                self.assertTrue(hasattr(j2, f))

                recur(getattr(j1, f), getattr(j2, f), path[f])

        elif isinstance(j1, pydantic.v1.BaseModel):
            for f in j1.__fields__:
                if f in skips:
                    continue

                self.assertTrue(hasattr(j2, f))

                recur(getattr(j1, f), getattr(j2, f), path[f])

        else:
            raise RuntimeError(
                f"Don't know how to compare objects of type {type(j1)} at {ps}."
            )
