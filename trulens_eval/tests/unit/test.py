from dataclasses import fields
from dataclasses import is_dataclass
from datetime import datetime
import functools
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Set, TypeVar
import unittest
from unittest import TestCase

from frozendict import frozendict
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


T = TypeVar("T")


def hashable_skip(obj: T, skips: Set[str]) -> T:
    """Return a hashable copy of `obj` with all keys/attributes in `skips`
    removed. 
    
    Sequences are returned as tuples and container types are returned as
    frozendicts. Floats are retuned as 0.0 to avoid tolerance issues. Note that
    the returned objects are only used for ordering their originals and are not
    compared themselves.

    Args:
        obj: The object to remove keys/attributes from.

        skips: The keys to remove.
    """

    def recur(obj):
        return hashable_skip(obj, skips=skips)

    if isinstance(obj, float):
        return 0.0

    if isinstance(obj, JSON_BASES):
        return obj

    if isinstance(obj, Mapping):
        return frozendict(
            {k: recur(v) for k, v in obj.items() if k not in skips}
        )

    elif isinstance(obj, Sequence):
        return tuple(recur(v) for v in obj)

    elif is_dataclass(obj):
        ret = {}
        for f in fields(obj):
            if f.name in skips:
                continue

            ret[f.name] = recur(getattr(obj, f.name))

        return frozendict(ret)

    elif isinstance(obj, BaseModel):
        ret = {}
        for f in obj.model_fields:
            if f in skips:
                continue

            ret[f] = recur(getattr(obj, f))

        return frozendict(ret)

    elif isinstance(obj, pydantic.v1.BaseModel):
        ret = {}

        for f in j1.__fields__:
            if f in skips:
                continue

            ret[f] = recur(getattr(obj, f))

        return frozendict(ret)

    else:
        return obj


def str_sorted(seq: Sequence[T], skips: Set[str]) -> Sequence[T]:
    """Return a sorted version of `obj` by string order.

    Items are convered to strings using `hashable_skip` with `skips`
    keys/attributes skipped.

    Args:
        seq: The sequence to sort.

        skips: The keys/attributes to skip for string conversion.
    """

    objs_and_strs = [(o, str(hashable_skip(o, skips=skips))) for o in seq]
    objs_and_strs_sorted = sorted(objs_and_strs, key=lambda x: x[1])

    return [o for o, _ in objs_and_strs_sorted]


class JSONTestCase(TestCase):
    """TestCase class that adds JSON comparisons and golden expectation handling."""

    def assertGoldenJSONEqual(
        self,
        actual: JSON,
        golden_filename: str,
        skips: Optional[Set[str]] = None,
        numeric_places: int = 7,
        unordereds: Optional[Set[str]] = None,
        unordered: bool = False,
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
                stores the expected JSON-like results for the test. File must
                have an extension of either `.json` or `.yaml`. The extension
                determines output format.

                !!! WARNING
                    YAML dumper does not fully serialize all types which
                    prevents them from being loaded again.            

            skips: A set of keys to skip in the comparison.

            numeric_places: The number of decimal places to compare for floating
                point

            unordereds: A set of keys or attribute names whose associated values
                are compared without orderered if they are sequences.

            unordered: If true, the order of elements in a sequence is not
                checked. Note that this only applies to the inputs `j1` and `j2`
                and not to any nested elements.

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
            writer = functools.partial(json.dump, indent=2)
            loader = json.load
        elif golden_path.suffix == ".yaml":
            writer = yaml.dump
            loader = functools.partial(yaml.load, Loader=yaml.FullLoader)
        else:
            raise ValueError(f"Unknown file extension {golden_path.suffix}.")

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
                actual,
                expected,
                skips=skips,
                numeric_places=numeric_places,
                unordereds=unordereds,
                unordered=unordered
            )

    def assertJSONEqual(
        self,
        j1: JSON,
        j2: JSON,
        path: Optional[Lens] = None,
        skips: Optional[Set[str]] = None,
        numeric_places: int = 7,
        unordereds: Optional[Set[str]] = None,
        unordered: bool = False,
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

            unordereds: A set of keys or attribute names whose associated values
                are compared without orderered if they are sequences.

            unordered: If true, the order of elements in a sequence is not
                checked. Note that this only applies to the inputs `j1` and `j2`
                and not to any nested elements.

        Raises:
            AssertionError: If the two JSON-like objects are
                not equal (except for anything skipped) or anything within
                numeric tolerance.
        """

        skips = skips or set([])
        path = path or Lens()
        unordereds = unordereds or set([])

        def recur(j1, j2, path, unordered=False):
            return self.assertJSONEqual(
                j1,
                j2,
                path=path,
                skips=skips,
                numeric_places=numeric_places,
                unordered=unordered,
                unordereds=unordereds
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

                recur(j1[k], j2[k], path=path[k], unordered=k in unordereds)

        elif isinstance(j1, Sequence):
            self.assertEqual(len(j1), len(j2), ps)

            if unordered:
                j1 = str_sorted(j1, skips=skips)
                j2 = str_sorted(j2, skips=skips)

            for i, (v1, v2) in enumerate(zip(j1, j2)):
                recur(v1, v2, path=path[i])

        elif isinstance(j1, datetime):
            self.assertEqual(j1, j2, ps)

        elif is_dataclass(j1):
            for f in fields(j1):
                if f.name in skips:
                    continue

                self.assertTrue(hasattr(j2, f.name))

                recur(
                    getattr(j1, f.name),
                    getattr(j2, f.name),
                    path[f.name],
                    unordered=f.name in unordereds
                )

        elif isinstance(j1, BaseModel):
            for f in j1.model_fields:
                if f in skips:
                    continue

                self.assertTrue(hasattr(j2, f))

                recur(
                    getattr(j1, f),
                    getattr(j2, f),
                    path[f],
                    unordered=f in unordereds
                )

        elif isinstance(j1, pydantic.v1.BaseModel):
            for f in j1.__fields__:
                if f in skips:
                    continue

                self.assertTrue(hasattr(j2, f))

                recur(
                    getattr(j1, f),
                    getattr(j2, f),
                    path[f],
                    unordered=f in unordereds
                )

        else:
            raise RuntimeError(
                f"Don't know how to compare objects of type {type(j1)} at {ps}."
            )
