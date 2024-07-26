from dataclasses import fields
from dataclasses import is_dataclass
from datetime import datetime
import os
from typing import Dict, Sequence
import unittest
from unittest import TestCase

from pydantic import BaseModel
from pydantic.v1 import BaseModel as v1BaseModel
from trulens.utils.serial import JSON_BASES
from trulens.utils.serial import Lens

# Env var that were to evaluate to true indicates that optional tests are to be
# run.
OPTIONAL_ENV_VAR = "TEST_OPTIONAL"


def optional_test(testmethodorclass):
    """
    Only run the decorated test if the environment variable with_optional
    evalutes true. These are meant to be run only in an environment where
    optional packages have been installed.
    """

    return unittest.skipIf(
        not os.environ.get(OPTIONAL_ENV_VAR), "optional test"
    )(testmethodorclass)


def requiredonly_test(testmethodorclass):
    """
    Only runs the decorated test if the environment variable with_optional
    evalutes to false or is not set. Decorated tests are meant to run
    specifically when optional imports are not installed.
    """

    return unittest.skipIf(
        os.environ.get(OPTIONAL_ENV_VAR), "not an optional test"
    )(testmethodorclass)


def module_installed(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


class JSONTestCase(TestCase):
    def assertJSONEqual(
        self, j1, j2, path: Lens = None, skips=None, numeric_places: int = 7
    ) -> None:
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

        elif isinstance(j1, v1BaseModel):
            for f in j1.__fields__:
                if f in skips:
                    continue

                self.assertTrue(hasattr(j2, f))

                recur(getattr(j1, f), getattr(j2, f), path[f])

        else:
            raise RuntimeError(
                f"Don't know how to compare objects of type {type(j1)} at {ps}."
            )
