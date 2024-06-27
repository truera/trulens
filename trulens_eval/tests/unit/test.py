from dataclasses import fields
from dataclasses import is_dataclass
from datetime import datetime
import json
import yaml
import os
from pathlib import Path
from typing import Dict, Sequence
import unittest
from unittest import TestCase

import pydantic
from pydantic import BaseModel

from trulens_eval.utils.python import caller_frame
from trulens_eval.utils.serial import JSON_BASES
from trulens_eval.utils.serial import Lens

# Env var that were to evaluate to true indicates that optional tests are to be
# run.
OPTIONAL_ENV_VAR = "TEST_OPTIONAL"


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
        actual,
        golden_filename: str,
        skips=None,
        numeric_places: int = 7,
    ):
        """Assert equality between JSON-like `actual` and the content of
        `golden_filename`.

        If the environment variable `WRITE_GOLDEN` is set, the golden file will
        be overwritten with the `actual` content. See `assertJSONEqual` for
        details on the equality check.
        """

        write_golden: bool = bool(os.environ.get("WRITE_GOLDEN", ""))

        caller_path = Path(caller_frame(offset=1).f_code.co_filename).parent
        golden_path = (caller_path / "golden" / golden_filename).resolve()

        if write_golden:
            with golden_path.open("w") as f:
                if golden_path.suffix == ".json":
                    json.dump(actual, f)
                elif golden_path.suffix == ".yaml":
                    yaml.dump(actual, f)
                else:
                    raise ValueError(f"Unknown file extension {golden_path.suffix}.")

            self.fail("Golden file written.")

        else:
            if not golden_path.exists():
                raise FileNotFoundError(f"Golden file {golden_path} not found.")

            if golden_path.suffix == ".json":
                with golden_path.open("r") as f:
                    expected = json.load(f)
            elif golden_path.suffix == ".yaml":
                with golden_path.open("r") as f:
                    expected = yaml.load(f, Loader=yaml.FullLoader)
            else:
                raise ValueError(f"Unknown file extension {golden_path.suffix}.")

            self.assertJSONEqual(
                actual, expected, skips=skips, numeric_places=numeric_places
            )

    def assertJSONEqual(
        self,
        j1,
        j2,
        path: Lens = None,
        skips=None,
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

        - int
        - float
        - str
        - dict
        - list
        - datetime
        - dataclasses
        - pydantic models
        
        Args:
            j1: The first JSON-like object.

            j2: The second JSON-like object.
            
            path: The path to the current object in the JSON structure.

            skips: A set of keys to skip in the comparison.

            numeric_places: The number of decimal places to compare for floating
            point
                numbers.
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
