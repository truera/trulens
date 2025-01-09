"""Tests for signature.py utilities."""

from inspect import signature
from unittest import TestCase
from unittest import main

from pydantic import BaseModel
from trulens.core.utils.signature import main_input
from trulens.core.utils.signature import main_output


class MockModel(BaseModel):
    content: str
    choices: list = []


class MockChoice(BaseModel):
    message: MockModel


class TestSignatureUtils(TestCase):
    def testMainInput(self):
        with self.subTest("Single String"):

            def func_string(a):
                return a

            sig = signature(func_string)
            bindings = sig.bind("test input")
            self.assertEqual(
                main_input(func_string, sig, bindings), "test input"
            )

        with self.subTest("Single Number"):

            def func_number(a):
                return a

            sig = signature(func_number)
            bindings = sig.bind(123)
            self.assertEqual(main_input(func_number, sig, bindings), "123")

        with self.subTest("Pydantic Model"):

            def func_model(a):
                return a

            sig = signature(func_model)
            model = MockModel(content="test content")
            bindings = sig.bind(model)
            self.assertEqual(
                main_input(func_model, sig, bindings), "test content"
            )

        with self.subTest("List of Strings"):

            def func_list(a):
                return a

            sig = signature(func_list)
            bindings = sig.bind(["test1", "test2"])
            self.assertEqual(main_input(func_list, sig, bindings), "test1")

        with self.subTest("List of Different Types"):

            def func_list_different_types(a):
                return a

            sig = signature(func_list_different_types)
            bindings = sig.bind(["test", 123, 45.6])
            self.assertEqual(
                main_input(func_list_different_types, sig, bindings), "test"
            )

        with self.subTest("Keyword Arguments"):

            def func_kwargs(a, b=None):
                return a if b is None else b

            sig = signature(func_kwargs)
            bindings = sig.bind("test input", b="kwarg input")
            self.assertEqual(
                main_input(func_kwargs, sig, bindings), "test input"
            )

        with self.subTest("Dict Input"):

            def func_dict(a):
                return a

            sig = signature(func_dict)
            bindings = sig.bind({"key1": "value1", "key2": "value2"})
            self.assertTrue(
                main_input(func_dict, sig, bindings).startswith(
                    "TruLens: Could not determine main input from "
                )
            )

    def testMainOutput(self):
        with self.subTest("String"):

            def func_string():
                return "test output"

            ret = func_string()
            self.assertEqual(main_output(func_string, ret), "test output")

        with self.subTest("List of Strings"):

            def func_list():
                return ["part1", "part2"]

            ret = func_list()
            self.assertEqual(main_output(func_list, ret), "part1part2")

        with self.subTest("Pydantic Model"):

            def func_model():
                return MockModel(content="test content")

            ret = func_model()
            self.assertEqual(main_output(func_model, ret), "test content")

        with self.subTest("List of Different Types"):

            def func_list_different_types():
                return ["test", 123, 45.6]

            ret = func_list_different_types()
            self.assertEqual(
                main_output(func_list_different_types, ret), "test"
            )

        with self.subTest("Keyword Arguments"):

            def func_kwargs(a, b=None):
                return a if b is None else b

            ret = func_kwargs("test output", b="kwarg output")
            self.assertEqual(main_output(func_kwargs, ret), "kwarg output")

        with self.subTest("Dict Output"):

            def func_dict():
                return {"key1": "value1", "key2": "value2"}

            ret = func_dict()
            self.assertEqual(main_output(func_dict, ret), "value1")


if __name__ == "__main__":
    main()
