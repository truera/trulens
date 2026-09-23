"""Unit tests for SingletonPerNameMeta and PydanticSingleton deletion."""

import unittest
from unittest import TestCase

import pydantic
from trulens.core.utils.python import PydanticSingleton
from trulens.core.utils.python import SingletonPerNameMeta


class SimpleSingleton(metaclass=SingletonPerNameMeta):
    pass


class NamedSingleton(metaclass=SingletonPerNameMeta):
    pass


class ModelSingleton(PydanticSingleton, pydantic.BaseModel):
    value: int = 1


class TestSingletonDeletion(TestCase):
    def test_delete_singleton_by_class(self):
        inst1 = SimpleSingleton()
        self.assertIn(
            (f"{SimpleSingleton.__module__}.{SimpleSingleton.__name__}", None),
            SingletonPerNameMeta._singleton_instances,
        )
        SingletonPerNameMeta.delete_singleton(SimpleSingleton)
        self.assertNotIn(
            (f"{SimpleSingleton.__module__}.{SimpleSingleton.__name__}", None),
            SingletonPerNameMeta._singleton_instances,
        )
        inst2 = SimpleSingleton()
        self.assertIsNot(inst1, inst2)
        SingletonPerNameMeta.delete_singleton(SimpleSingleton)

    def test_delete_singleton_by_instance(self):
        inst1 = SimpleSingleton()
        SingletonPerNameMeta.delete_singleton(inst1)
        self.assertNotIn(
            (f"{SimpleSingleton.__module__}.{SimpleSingleton.__name__}", None),
            SingletonPerNameMeta._singleton_instances,
        )

    def test_delete_singleton_with_name(self):
        NamedSingleton(name="alpha")
        NamedSingleton(name="beta")
        key_a = (
            f"{NamedSingleton.__module__}.{NamedSingleton.__name__}",
            "alpha",
        )
        key_b = (
            f"{NamedSingleton.__module__}.{NamedSingleton.__name__}",
            "beta",
        )
        self.assertIn(key_a, SingletonPerNameMeta._singleton_instances)
        self.assertIn(key_b, SingletonPerNameMeta._singleton_instances)

        SingletonPerNameMeta.delete_singleton(NamedSingleton, name="alpha")
        self.assertNotIn(key_a, SingletonPerNameMeta._singleton_instances)
        self.assertIn(key_b, SingletonPerNameMeta._singleton_instances)

        SingletonPerNameMeta.delete_singleton(NamedSingleton, name="beta")
        self.assertNotIn(key_b, SingletonPerNameMeta._singleton_instances)

    def test_pydantic_singleton_deletion(self):
        m1 = ModelSingleton(value=42)
        self.assertIn(
            (f"{ModelSingleton.__module__}.{ModelSingleton.__name__}", None),
            SingletonPerNameMeta._singleton_instances,
        )
        ModelSingleton.delete_singleton(ModelSingleton)
        self.assertNotIn(
            (f"{ModelSingleton.__module__}.{ModelSingleton.__name__}", None),
            SingletonPerNameMeta._singleton_instances,
        )
        m2 = ModelSingleton(value=99)
        self.assertEqual(m2.value, 99)
        self.assertIsNot(m1, m2)
        ModelSingleton.delete_singleton(ModelSingleton)


if __name__ == "__main__":
    unittest.main()
