"""Tests for endpoint module."""

from enum import Enum

from trulens.core.feedback.endpoint import Endpoint


class TestWrapFunction:
    """Tests for Endpoint.wrap_function handling of various input types."""

    def test_wrap_function_skips_non_callable_enum(self):
        """Test that wrap_function skips enum values without raising."""

        class MockCallTypes(Enum):
            completion = "completion"
            chat = "chat"

        endpoint = Endpoint(name="test-endpoint")

        # This should not raise AttributeError
        result = endpoint.wrap_function(MockCallTypes.completion)

        # Should return the original enum value unchanged
        assert result is MockCallTypes.completion

    def test_wrap_function_skips_non_callable_string(self):
        """Test that wrap_function skips string values."""
        endpoint = Endpoint(name="test-endpoint")

        result = endpoint.wrap_function("not a function")

        assert result == "not a function"

    def test_wrap_function_skips_non_callable_int(self):
        """Test that wrap_function skips integer values."""
        endpoint = Endpoint(name="test-endpoint")

        result = endpoint.wrap_function(42)

        assert result == 42

    def test_wrap_function_wraps_callable(self):
        """Test that wrap_function properly wraps callable functions."""
        endpoint = Endpoint(name="test-endpoint")

        def sample_function():
            return "result"

        wrapped = endpoint.wrap_function(sample_function)

        # The wrapped function should be callable
        assert callable(wrapped)
        # Should have the INSTRUMENT attribute set (uses __tru_instrument internally)
        assert hasattr(wrapped, "__tru_instrument")

    def test_wrap_function_wraps_lambda(self):
        """Test that wrap_function wraps lambda functions."""
        endpoint = Endpoint(name="test-endpoint")

        func = lambda x: x * 2

        wrapped = endpoint.wrap_function(func)

        assert callable(wrapped)


class TestInstrumentClass:
    """Tests for Endpoint._instrument_class handling of enum classes."""

    def test_instrument_class_skips_enum_classes(self):
        """Test that _instrument_class skips Enum classes without raising."""

        class MockCallTypes(Enum):
            completion = "completion"
            chat = "chat"

        endpoint = Endpoint(name="test-endpoint")

        # This should not raise AttributeError about reassigning enum member
        endpoint._instrument_class(MockCallTypes, "completion")

        # The enum should be unchanged
        assert MockCallTypes.completion.value == "completion"

    def test_instrument_class_works_on_regular_classes(self):
        """Test that _instrument_class still works on regular classes."""

        class MockClass:
            def completion(self):
                return "result"

        endpoint = Endpoint(name="test-endpoint")
        endpoint._instrument_class(MockClass, "completion")

        # The method should be instrumented
        assert hasattr(MockClass.completion, "__tru_instrument")
