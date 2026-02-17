"""Tests for SerialModel repr with circular references.

Reproduces the RecursionError described in GitHub issue #1862:
pydantic's default __repr__ does not handle circular references
among model instances, leading to infinite recursion.
"""

from trulens.core.utils import serial as serial_utils


class ModelA(serial_utils.SerialModel):
    """A model that can hold a reference to ModelB."""

    name: str = "a"
    other: object = None

    model_config = {"arbitrary_types_allowed": True}


class ModelB(serial_utils.SerialModel):
    """A model that can hold a reference back to ModelA."""

    name: str = "b"
    other: object = None

    model_config = {"arbitrary_types_allowed": True}


class TestSerialModelRepr:
    """Verify that repr() on SerialModel handles circular refs."""

    def test_repr_circular_reference_does_not_recurse(self):
        """Two SerialModel instances referencing each other must not
        cause a RecursionError when repr() is called.

        This is the core bug from issue #1862.
        """
        a = ModelA()
        b = ModelB()
        a.other = b
        b.other = a

        # Before the fix this raises RecursionError.
        result = repr(a)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_repr_self_reference_does_not_recurse(self):
        """A model referencing itself must not cause RecursionError."""
        a = ModelA()
        a.other = a

        result = repr(a)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_repr_no_circular_reference_still_works(self):
        """Normal (non-circular) models must still repr correctly."""
        a = ModelA(name="hello")
        result = repr(a)
        assert "hello" in result

    def test_repr_deep_chain_does_not_recurse(self):
        """A -> B -> A -> B ... chain must be handled."""
        a = ModelA(name="root")
        b = ModelB(name="child")
        a.other = b
        b.other = ModelA(name="grandchild", other=a)

        result = repr(a)
        assert isinstance(result, str)

    def test_str_on_record_does_not_recurse(self):
        """Record.__str__ (and by extension __repr__) should be safe
        even when the record contains complex nested objects.
        """
        from trulens.core.schema import record as record_schema

        record = record_schema.Record(
            app_id="test_app",
            calls=[],
        )
        # str() and repr() should both work without error.
        assert isinstance(str(record), str)
        assert isinstance(repr(record), str)

    def test_repr_otel_event_with_circular_reference(self):
        """OTEL Event (SerialModel subclass) must handle circular
        references in its dict-typed fields without RecursionError.
        """
        from datetime import datetime

        from trulens.core.schema import event as event_schema

        event = event_schema.Event(
            event_id="evt-1",
            record={"name": "test_func", "kind": "SPAN_KIND_TRULENS"},
            record_attributes={"key": "value"},
            record_type=event_schema.EventRecordType.SPAN,
            resource_attributes={},
            start_timestamp=datetime.now(),
            timestamp=datetime.now(),
            trace={
                "trace_id": "t1",
                "parent_id": "p1",
                "span_id": "s1",
            },
        )
        # Normal case should work fine.
        result = repr(event)
        assert "evt-1" in result

        # Inject a circular reference into a dict field.
        circular: dict = {}
        circular["self"] = circular
        event.record_attributes = {"nested": circular}

        # Must not raise RecursionError.
        result = repr(event)
        assert isinstance(result, str)

    def test_repr_otel_events_referencing_each_other(self):
        """Two OTEL Event objects cross-referencing each other via
        their dict fields must not cause RecursionError.
        """
        from datetime import datetime

        from trulens.core.schema import event as event_schema

        now = datetime.now()
        base = dict(
            record={"name": "fn"},
            record_type=event_schema.EventRecordType.SPAN,
            resource_attributes={},
            start_timestamp=now,
            timestamp=now,
            trace={
                "trace_id": "t",
                "parent_id": "p",
                "span_id": "s1",
            },
        )
        evt_a = event_schema.Event(event_id="a", record_attributes={}, **base)
        evt_b = event_schema.Event(
            event_id="b",
            record_attributes={},
            **{**base, "trace": {**base["trace"], "span_id": "s2"}},
        )
        # Cross-reference the two events.
        evt_a.record_attributes["related"] = evt_b
        evt_b.record_attributes["related"] = evt_a

        result = repr(evt_a)
        assert isinstance(result, str)
        assert len(result) > 0
