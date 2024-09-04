"""TestCase subclass with JSON comparisons and test enable/disable flag
handling."""

import asyncio
from dataclasses import fields
from dataclasses import is_dataclass
from datetime import datetime
import functools
import importlib
import json
import os
from pathlib import Path
from typing import (
    Dict,
    Mapping,
    Optional,
    OrderedDict,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import unittest
from unittest import TestCase

import pydantic
from pydantic import BaseModel
from trulens.core.utils.serial import JSON
from trulens.core.utils.serial import JSON_BASES
from trulens.core.utils.serial import Lens
import yaml

OPTIONAL_ENV_VAR = "TEST_OPTIONAL"
"""Env var that were to evaluate to true indicates that optional tests are to be
run."""

WRITE_GOLDEN_VAR = "WRITE_GOLDEN"
"""Env var for indicating whether golden expected results are to be written (if
true) or read and compared (if false/undefined)."""

ALLOW_OPTIONAL_ENV_VAR = "ALLOW_OPTIONALS"


def async_test(func):
    """Decorator for running async tests."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        temp = loop.run_until_complete(func(*args, **kwargs))
        loop.close()
        return temp

    return wrapper


def optional_test(testmethodorclass):
    """Only run the decorated test if the environment variable with_optional
    evaluates true.

    These are meant to be run only in an environment where
    all optional packages have been installed.
    """

    return unittest.skipIf(
        not os.environ.get(OPTIONAL_ENV_VAR), "optional test"
    )(testmethodorclass)


def requiredonly_test(testmethodorclass):
    """Only runs the decorated test if the environment variable with_optional
    evaluates to false or is not set.

    Decorated tests are meant to run specifically when optional imports are not
    installed. ALLOW_EXTRA_DEPS allows optional imports to be installed.
    """

    return unittest.skipIf(
        os.environ.get(OPTIONAL_ENV_VAR)
        or os.environ.get(ALLOW_OPTIONAL_ENV_VAR),
        "not an optional test",
    )(testmethodorclass)


def module_installed(module: str) -> bool:
    """Check if a module is installed."""

    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


T = TypeVar("T")


def canonical(obj: T, skips: Set[str]) -> Union[T, Dict, Tuple]:
    """Return a canonical copy of `obj` with all keys/attributes in `skips`
    removed.

    Dicts are returned as OrderedDict with sorted keys. Keys in `skips` are
    omitted. Sequences are returned as Tuples no matter what sequence type the
    input was. Floats are returned as 0.0 to avoid tolerance issues. Note that
    the returned objects are only used for ordering their originals and are not
    compared themselves.

    Args:
        obj: The object to remove keys/attributes from.

        skips: The keys to remove.
    """

    def recur(obj):
        return canonical(obj, skips=skips)

    if isinstance(obj, float):
        return 0.0

    if isinstance(obj, JSON_BASES):
        return obj

    if isinstance(obj, Mapping):
        ret = OrderedDict()
        for k in sorted(obj.keys()):
            v = obj[k]
            if k in skips:
                continue
            ret[k] = recur(v)

        return ret

    elif isinstance(obj, Sequence):
        return tuple(recur(v) for v in obj)

    elif is_dataclass(obj):
        ret = OrderedDict()

        for f in sorted(fields(obj), key=lambda f: f.name):
            if f.name in skips:
                continue

            ret[f.name] = recur(getattr(obj, f.name))

        return ret

    elif isinstance(obj, BaseModel):
        ret = OrderedDict()
        for f in sorted(obj.model_fields):
            if f in skips:
                continue

            ret[f] = recur(getattr(obj, f))

        return ret

    elif isinstance(obj, pydantic.v1.BaseModel):
        ret = OrderedDict()

        for f in sorted(obj.__fields__):
            if f in skips:
                continue

            ret[f] = recur(getattr(obj, f))

        return ret

    else:
        raise TypeError(f"Unhandled type {type(obj).__name__}.")


def str_sorted(seq: Sequence[T], skips: Set[str]) -> Sequence[T]:
    """Return a sorted version of `seq` by string order.

    Items are converted to strings using `canonical` with `skips`
    keys/attributes skipped.

    Args:
        seq: The sequence to sort.

        skips: The keys/attributes to skip for string conversion.
    """

    objs_and_strs = [(o, str(canonical(o, skips=skips))) for o in seq]
    objs_and_strs_sorted = sorted(objs_and_strs, key=lambda x: x[1])

    return [o for o, _ in objs_and_strs_sorted]


class WithJSONTestCase(TestCase):
    """TestCase mixin class that adds JSON comparisons and golden expectation
    handling."""

    def load_golden(self, golden_path: Union[str, Path]) -> JSON:
        """Load the golden file `path` and return its contents.

        Args:
            golden_path: The name of the golden file to load. The file must
                have an extension of either `.json` or `.yaml`. The extension
                determines the input format.

        """
        golden_path = Path(golden_path)

        if ".json" in golden_path.suffixes:
            loader = functools.partial(json.load)
        elif ".yaml" in golden_path.suffixes or ".yml" in golden_path.suffixes:
            loader = functools.partial(yaml.load, Loader=yaml.FullLoader)
        else:
            raise ValueError(f"Unknown file extension {golden_path}.")

        if not golden_path.exists():
            raise FileNotFoundError(f"Golden file {golden_path} not found.")

        with golden_path.open() as f:
            return loader(f)

    def write_golden(self, golden_path: Union[str, Path], data: JSON) -> None:
        """If writing golden file is enabled, write the golden file `path` with
        `data` and raise exception indicating so.

        If not writing golden file, does nothing.

        Args:
            golden_path: The path to the golden file to write. Format is
                determined by suffix.

            data: The data to write to the golden file.
        """
        if not self.writing_golden():
            return

        golden_path = Path(golden_path)

        if golden_path.suffix == ".json":
            writer = functools.partial(json.dump, indent=2, sort_keys=True)
        elif golden_path.suffix == ".yaml":
            writer = functools.partial(yaml.dump, sort_keys=True)
        else:
            raise ValueError(f"Unknown file extension {golden_path.suffix}.")

        with golden_path.open("w") as f:
            writer(data, f)

        self.fail(f"Golden file {golden_path} written.")

    def writing_golden(self) -> bool:
        """Return whether the golden files are to be written."""

        return bool(os.environ.get(WRITE_GOLDEN_VAR, ""))

    def assertGoldenJSONEqual(
        self,
        actual: JSON,
        golden_path: Union[Path, str],
        skips: Optional[Set[str]] = None,
        numeric_places: int = 7,
        unordereds: Optional[Set[str]] = None,
        unordered: bool = False,
    ):
        """Assert equality between [JSON-like][trulens_eval.util.serial.JSON]
        `actual` and the content of `golden_path`.

        If the environment variable
        [WRITE_GOLDEN_VAR][trulens_eval.tests.unit.test.WRITE_GOLDEN_VAR] is
        set, the golden file will be overwritten with the `actual` content. See
        [assertJSONEqual][trulens_eval.tests.unit.test.assertJSONEqual] for
        details on the equality check.

        Args:
            actual: The actual JSON-like object produced by some test.

            golden_path: The path to the golden file to compare against that
                stores the expected JSON-like results for the test. File must
                have an extension of either `.json` or `.yaml`. The extension
                determines output format. This is in relation to the git base
                directory.

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

        if isinstance(golden_path, str):
            golden_path = Path(golden_path)

        # Write golden and raise exception if writing golden is enabled.
        self.write_golden(golden_path=golden_path, data=actual)

        # Otherwise load the golden file and compare.
        expected = self.load_golden(golden_path)

        self.assertJSONEqual(
            actual,
            expected,
            skips=skips,
            numeric_places=numeric_places,
            unordereds=unordereds,
            unordered=unordered,
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
                unordereds=unordereds,
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

            with self.subTest("keys", path=ps):
                self.assertSetEqual(ks1, ks2, ps)

            for k in ks1:
                if k in skips or k not in ks2:
                    continue

                with self.subTest(k, path=ps):
                    recur(j1[k], j2[k], path=path[k], unordered=k in unordereds)

        elif isinstance(j1, Sequence):
            self.assertEqual(len(j1), len(j2), ps)

            if unordered:
                j1 = str_sorted(j1, skips=skips)
                j2 = str_sorted(j2, skips=skips)

            for i, (v1, v2) in enumerate(zip(j1, j2)):
                with self.subTest(i, path=ps):
                    recur(v1, v2, path=path[i])

        elif isinstance(j1, datetime):
            self.assertEqual(j1, j2, ps)

        elif is_dataclass(j1):
            for f in fields(j1):
                if f.name in skips:
                    continue

                with self.subTest(f.name, path=ps):
                    self.assertTrue(hasattr(j2, f.name))

                    recur(
                        getattr(j1, f.name),
                        getattr(j2, f.name),
                        path[f.name],
                        unordered=f.name in unordereds,
                    )

        elif isinstance(j1, BaseModel):
            for f in j1.model_fields:
                if f in skips:
                    continue

                with self.subTest(f, path=ps):
                    self.assertTrue(hasattr(j2, f))

                    recur(
                        getattr(j1, f),
                        getattr(j2, f),
                        path[f],
                        unordered=f in unordereds,
                    )

        elif isinstance(j1, pydantic.v1.BaseModel):
            for f in j1.__fields__:
                if f in skips:
                    continue

                with self.subTest(f, path=ps):
                    self.assertTrue(hasattr(j2, f))

                    recur(
                        getattr(j1, f),
                        getattr(j2, f),
                        path[f],
                        unordered=f in unordereds,
                    )

        else:
            raise RuntimeError(
                f"Don't know how to compare objects of type {type(j1)} at {ps}."
            )


class JSONTestCase(WithJSONTestCase, TestCase):
    """TestCase subclass with JSON comparisons and test enable/disable flag
    handling."""

    pass
