from __future__ import annotations

import json
import logging
from typing import Any, cast

import pydantic
from trulens.core.utils import imports as import_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import serial as serial_utils

with import_utils.OptionalImports(
    messages=import_utils.format_import_errors(
        "jsonschema", purpose="validating output against a JSON schema dict"
    )
) as _jsonschema_opt:
    import jsonschema

logger = logging.getLogger(__name__)

_SchemaType = dict[str, Any] | type[pydantic.BaseModel]


class SchemaValidator(pyschema_utils.WithClassInfo, serial_utils.SerialModel):
    """Non-LLM feedback functions for validating LLM output against a schema.

    Accepts either a JSON schema dict (requires `jsonschema`) or a Pydantic
    model class.  Each method returns `1.0` when the output is valid and `0.0`
    otherwise, along with a metadata dict that contains any validation errors.

    Example — JSON schema dict:
        ```python
        from trulens.feedback.schema_validator import SchemaValidator
        from trulens.core.metric.metric import Metric

        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        validator = SchemaValidator(schema=schema)
        f = Metric(validator.validate_json).on_output()
        ```

    Example — Pydantic model:
        ```python
        import pydantic
        from trulens.feedback.schema_validator import SchemaValidator
        from trulens.core.metric.metric import Metric

        class MyOutput(pydantic.BaseModel):
            answer: str
            score: float

        validator = SchemaValidator(schema=MyOutput)
        f = Metric(validator.validate_json).on_output()
        ```
    """

    _schema: _SchemaType = pydantic.PrivateAttr()

    model_config: pydantic.ConfigDict = pydantic.ConfigDict(
        arbitrary_types_allowed=True
    )

    def __init__(self, schema: _SchemaType, **kwargs):
        """Create a SchemaValidator.

        Args:
            schema: Either a JSON schema dict or a Pydantic ``BaseModel``
                *class* (not an instance).
        """
        super().__init__(**kwargs)
        if not (
            isinstance(schema, dict)
            or (
                isinstance(schema, type)
                and issubclass(schema, pydantic.BaseModel)
            )
        ):
            raise TypeError(
                "`schema` must be a JSON schema dict or a Pydantic BaseModel class."
            )
        self._schema = schema

    def _validate_with_jsonschema(self, data: Any) -> tuple[bool, str | None]:
        _jsonschema_opt.assert_installed(jsonschema)
        try:
            jsonschema.validate(
                instance=data, schema=cast(dict[str, Any], self._schema)
            )
            return True, None
        except jsonschema.ValidationError as exc:
            return False, exc.message
        except jsonschema.SchemaError as exc:
            raise ValueError(
                f"Invalid JSON schema provided: {exc.message}"
            ) from exc

    def _validate_with_pydantic(self, data: Any) -> tuple[bool, str | None]:
        model_cls = cast(type[pydantic.BaseModel], self._schema)
        try:
            model_cls.model_validate(data)
            return True, None
        except pydantic.ValidationError as exc:
            errors = "; ".join(
                f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
                for e in exc.errors()
            )
            return False, errors

    def _parse_and_validate(self, output: str) -> tuple[float, dict[str, str]]:
        """Parse *output* as JSON and validate against the stored schema.

        Returns:
            Tuple of ``(score, metadata)`` where *score* is ``1.0`` if valid
            and ``0.0`` otherwise.  *metadata* always contains an
            ``"explanation"`` key.
        """
        try:
            data = json.loads(output)
        except (json.JSONDecodeError, TypeError) as exc:
            return 0.0, {"explanation": f"Output is not valid JSON: {exc}"}

        if isinstance(self._schema, dict):
            ok, error_msg = self._validate_with_jsonschema(data)
        else:
            ok, error_msg = self._validate_with_pydantic(data)

        if ok:
            return 1.0, {"explanation": "Output conforms to the schema."}
        return 0.0, {"explanation": f"Schema validation failed: {error_msg}"}

    def validate_json(self, output: str) -> tuple[float, dict[str, str]]:
        """Validate that *output* is valid JSON conforming to the schema.

        Returns ``1.0`` when valid, ``0.0`` otherwise.  The accompanying
        metadata dict always contains an ``"explanation"`` key describing the
        outcome (or the first validation error).

        Args:
            output: The string to validate.  It must be parseable as JSON.

        Returns:
            A ``(score, metadata)`` tuple compatible with TruLens feedback
            infrastructure.
        """
        return self._parse_and_validate(output)

    def validate_json_partial(
        self, output: str, required_keys: list | None = None
    ) -> tuple[float, dict[str, str]]:
        """Validate that *output* is valid JSON and optionally check for keys.

        This is a lighter-weight check: it verifies that *output* parses as a
        JSON object and, when *required_keys* is provided, that each key is
        present at the top level.  The full schema is not consulted, making
        this useful for streaming or partial outputs.

        Args:
            output: The string to validate.
            required_keys: Optional list of keys that must exist at the top
                level of the parsed object.

        Returns:
            A ``(score, metadata)`` tuple.
        """
        try:
            data = json.loads(output)
        except (json.JSONDecodeError, TypeError) as exc:
            return 0.0, {"explanation": f"Output is not valid JSON: {exc}"}

        if not isinstance(data, dict):
            return 0.0, {
                "explanation": "Output is valid JSON but not a JSON object."
            }

        if required_keys:
            missing = [k for k in required_keys if k not in data]
            if missing:
                return 0.0, {"explanation": f"Missing required keys: {missing}"}

        return (
            1.0,
            {
                "explanation": "Output is a valid JSON object with all required keys."
            },
        )
