from dataclasses import fields, is_dataclass
from typing import Dict, Sequence
from unittest import TestCase

from pydantic import BaseModel

from trulens_eval.util import JSON_BASES, JSONPath
from datetime import datetime


class JSONTestCase(TestCase):

    def assertJSONEqual(
        self,
        j1,
        j2,
        path: JSONPath = None,
        skips=None,
        numeric_places: int = 7
    ) -> None:
        skips = skips or set([])
        path = path or JSONPath()

        def recur(j1, j2, path):
            return self.assertJSONEqual(
                j1, j2, path=path, skips=skips, numeric_places=numeric_places
            )

        ps = str(path)

        self.assertIsInstance(j1, type(j2), ps)

        if isinstance(j1, JSON_BASES):
            if isinstance(j1, (int, float)):
                self.assertAlmostEqual(j1, j2, places=numeric_places)
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

                recur(
                    getattr(j1, f.name), getattr(j2, f.name),
                    getattr(path, f.name)
                )

        elif isinstance(j1, BaseModel):
            for f in j1.__fields__:
                if f in skips:
                    continue

                recur(getattr(j1, f), getattr(j2, f), getattr(path, f))

        else:
            raise RuntimeError(
                f"Don't know how to compare objects of type {type(j1)} at {ps}."
            )
