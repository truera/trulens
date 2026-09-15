"""Serializable prompt-management classes.

Prompts are reusable, versioned definitions stored in the configured TruLens
database. A [Prompt][trulens.core.schema.prompt.Prompt] is the stable identity,
a [PromptVersion][trulens.core.schema.prompt.PromptVersion] is one immutable
content-addressed revision of it, and a
[PromptLabel][trulens.core.schema.prompt.PromptLabel] is a mutable pointer from
a name such as `production` to one exact version.

Rendering happens locally and never calls a model.
"""

from __future__ import annotations

import datetime
from enum import Enum
import logging
import re
from typing import Any, Dict, Hashable, List, Optional, Sequence, Union

import pydantic
from trulens.core.schema import types as types_schema
from trulens.core.utils import json as json_utils
from trulens.core.utils import serial as serial_utils

logger = logging.getLogger(__name__)

VARIABLE_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
"""Pattern for a simple named variable such as `{{question}}`."""

_ANY_MUSTACHE_PATTERN = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)
"""Pattern for anything delimited by double braces, valid or not."""

_STATEMENT_PATTERN = re.compile(r"\{%.*?%\}", re.DOTALL)
"""Pattern for statement tags such as `{% for x in y %}`."""

RESERVED_RENDER_KWARGS = ("model_overrides", "strict")
"""Keyword arguments of the render methods that cannot name a variable."""

LATEST_LABEL = "latest"
"""Label moved onto every newly created version."""


class PromptType(str, Enum):
    """The kind of content a prompt holds.

    Fixed at creation time. A prompt cannot change from text to chat or back.
    """

    TEXT = "text"
    """One template string."""

    CHAT = "chat"
    """An ordered list of role-tagged messages."""


class MessageRole(str, Enum):
    """Role of a message in a chat prompt."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class PromptMessage(serial_utils.SerialModel):
    """One message of a chat prompt."""

    role: MessageRole
    """Who the message is attributed to."""

    content: Union[str, List[Dict[str, Any]]]
    """Message text, or a list of structured content blocks."""

    def content_variables(self) -> List[str]:
        """Variable names referenced by this message, in order of appearance."""

        return _variables_in(_content_strings(self.content))


class InvalidTemplateError(ValueError):
    """Raised when template content uses unsupported syntax."""


class VariableError(ValueError):
    """Raised when render values do not match the declared variables."""


def _content_strings(content: Union[str, Sequence[Any]]) -> List[str]:
    """Collect every string that interpolation should look at."""

    if isinstance(content, str):
        return [content]

    strings = []
    for block in content:
        if isinstance(block, str):
            strings.append(block)
        elif isinstance(block, dict):
            for value in block.values():
                if isinstance(value, str):
                    strings.append(value)
    return strings


def _variables_in(strings: Sequence[str]) -> List[str]:
    """Ordered, de-duplicated variable names found in `strings`."""

    found: List[str] = []
    for text in strings:
        for name in VARIABLE_PATTERN.findall(text):
            if name not in found:
                found.append(name)
    return found


def validate_template(strings: Sequence[str]) -> None:
    """Reject template syntax beyond simple named variables.

    Args:
        strings: Template strings to check.

    Raises:
        InvalidTemplateError: If a statement tag is present or a `{{ ... }}`
            span is not a bare variable name.
    """

    for text in strings:
        statement = _STATEMENT_PATTERN.search(text)
        if statement is not None:
            raise InvalidTemplateError(
                f"Statement tags are not supported: {statement.group(0)!r}. "
                "Prompt templates support simple named variables such as "
                "{{question}} only."
            )
        for match in _ANY_MUSTACHE_PATTERN.finditer(text):
            if VARIABLE_PATTERN.fullmatch(match.group(0)) is None:
                raise InvalidTemplateError(
                    f"Unsupported template expression: {match.group(0)!r}. "
                    "Prompt templates support simple named variables such as "
                    "{{question}} only."
                )


def _interpolate(text: str, values: Dict[str, Any]) -> str:
    return VARIABLE_PATTERN.sub(lambda m: str(values[m.group(1)]), text)


def _interpolate_content(
    content: Union[str, List[Dict[str, Any]]], values: Dict[str, Any]
) -> Union[str, List[Dict[str, Any]]]:
    if isinstance(content, str):
        return _interpolate(content, values)

    blocks = []
    for block in content:
        blocks.append({
            key: _interpolate(value, values)
            if isinstance(value, str)
            else value
            for key, value in block.items()
        })
    return blocks


class Prompt(serial_utils.SerialModel, Hashable):
    """The stable identity of a reusable prompt.

    The identifier is derived from the slug so that renaming the prompt or
    editing its description does not break existing references.
    """

    prompt_id: types_schema.PromptID  # str
    """The unique identifier for the prompt."""

    slug: str
    """Short, stable, human-written key such as `support-assistant`."""

    name: str
    """Display name."""

    prompt_type: PromptType
    """Whether versions of this prompt hold text or chat content.

    Immutable once the prompt exists.
    """

    description: Optional[str] = None
    """Free-text description."""

    tags: List[str] = pydantic.Field(default_factory=list)
    """Tags for grouping prompts."""

    created_at: datetime.datetime = pydantic.Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    """When the prompt was first created."""

    updated_at: datetime.datetime = pydantic.Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    """When the prompt metadata last changed."""

    def __init__(
        self,
        slug: str,
        name: Optional[str] = None,
        prompt_type: Union[PromptType, str] = PromptType.TEXT,
        prompt_id: Optional[types_schema.PromptID] = None,
        **kwargs,
    ):
        kwargs["slug"] = slug
        kwargs["name"] = name if name is not None else slug
        kwargs["prompt_type"] = PromptType(prompt_type)

        if prompt_id is None:
            prompt_id = json_utils.obj_id_of_obj(
                {"slug": slug}, prefix="prompt"
            )

        super().__init__(prompt_id=prompt_id, **kwargs)

    def __hash__(self):
        return hash(self.prompt_id)


class PromptVersion(serial_utils.SerialModel, Hashable):
    """One immutable revision of a prompt.

    The identifier is content-addressed over the prompt identity, the canonical
    content, the declared variables, the model defaults, and the response
    format, so creating the same version twice is idempotent.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    version_id: types_schema.PromptVersionID  # str
    """The unique identifier for this version."""

    prompt_id: types_schema.PromptID  # str
    """The prompt this version belongs to."""

    prompt_type: PromptType
    """Matches the type of the owning prompt."""

    parent_version_id: Optional[types_schema.PromptVersionID] = None
    """The version this one was derived from, when known."""

    text: Optional[str] = None
    """Template string. Set for text prompts only."""

    messages: Optional[List[PromptMessage]] = None
    """Ordered messages. Set for chat prompts only."""

    variables: List[str] = pydantic.Field(default_factory=list)
    """Names of the variables this version declares."""

    model_defaults: Dict[str, Any] = pydantic.Field(default_factory=dict)
    """Provider-neutral model settings persisted with the version.

    Never holds credentials.
    """

    response_format: Optional[Dict[str, Any]] = None
    """Response-format metadata handed to provider adapters."""

    change_note: Optional[str] = None
    """Why this version was created."""

    content_hash: str = ""
    """Hash of the canonical content alone."""

    created_at: datetime.datetime = pydantic.Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    """When the version was created."""

    created_by: Optional[str] = None
    """Who created the version, when the caller supplies it."""

    def __init__(
        self,
        prompt_id: types_schema.PromptID,
        prompt_type: Union[PromptType, str],
        text: Optional[str] = None,
        messages: Optional[
            Sequence[Union[PromptMessage, Dict[str, Any]]]
        ] = None,
        variables: Optional[Sequence[str]] = None,
        version_id: Optional[types_schema.PromptVersionID] = None,
        **kwargs,
    ):
        prompt_type = PromptType(prompt_type)

        parsed_messages: Optional[List[PromptMessage]] = None
        if messages is not None:
            parsed_messages = [
                message
                if isinstance(message, PromptMessage)
                else PromptMessage(**message)
                for message in messages
            ]

        if prompt_type is PromptType.TEXT:
            if text is None or parsed_messages is not None:
                raise ValueError(
                    "A text prompt version needs `text` and no `messages`."
                )
            strings = [text]
        else:
            if not parsed_messages or text is not None:
                raise ValueError(
                    "A chat prompt version needs `messages` and no `text`."
                )
            strings = [
                string
                for message in parsed_messages
                for string in _content_strings(message.content)
            ]

        validate_template(strings)

        found = _variables_in(strings)
        if variables is None:
            declared = found
        else:
            declared = list(variables)
            undeclared = [name for name in found if name not in declared]
            if undeclared:
                raise VariableError(
                    f"Template uses undeclared variables: {sorted(undeclared)}."
                )

        reserved = [name for name in declared if name in RESERVED_RENDER_KWARGS]
        if reserved:
            raise VariableError(
                "These variable names are reserved by the render methods and "
                f"cannot be used: {sorted(reserved)}."
            )

        content_hash = json_utils.obj_id_of_obj(
            {"prompt_type": prompt_type.value, "content": strings},
            prefix="content",
        )

        kwargs["prompt_id"] = prompt_id
        kwargs["prompt_type"] = prompt_type
        kwargs["text"] = text
        kwargs["messages"] = parsed_messages
        kwargs["variables"] = declared
        kwargs["content_hash"] = content_hash

        if version_id is None:
            version_id = json_utils.obj_id_of_obj(
                {
                    "prompt_id": prompt_id,
                    "prompt_type": prompt_type.value,
                    "content": strings,
                    "roles": [message.role.value for message in parsed_messages]
                    if parsed_messages
                    else None,
                    "variables": declared,
                    "model_defaults": kwargs.get("model_defaults") or {},
                    "response_format": kwargs.get("response_format"),
                },
                prefix="prompt_version",
            )

        super().__init__(version_id=version_id, **kwargs)

    def __hash__(self):
        return hash(self.version_id)


class PromptLabel(serial_utils.SerialModel):
    """A mutable pointer from a label to one exact version.

    `staging` and `production` are conventions, not hosted environments.
    """

    prompt_id: types_schema.PromptID  # str
    """The prompt this label belongs to."""

    label: str
    """The label name."""

    version_id: types_schema.PromptVersionID  # str
    """The exact version the label currently points at."""

    updated_at: datetime.datetime = pydantic.Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    """When the pointer last moved."""


class PromptLabelHistory(serial_utils.SerialModel):
    """One append-only record of a label movement."""

    history_id: str
    """The unique identifier for this history entry."""

    prompt_id: types_schema.PromptID  # str
    """The prompt whose label moved."""

    label: str
    """The label that moved."""

    previous_version_id: Optional[types_schema.PromptVersionID] = None
    """Where the label pointed before, or None on first assignment."""

    new_version_id: types_schema.PromptVersionID  # str
    """Where the label points after the move."""

    moved_by: Optional[str] = None
    """Caller label supplied by whoever moved the pointer."""

    timestamp: datetime.datetime = pydantic.Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    """When the move happened."""


class RenderedPrompt(serial_utils.SerialModel):
    """The local result of rendering one exact version.

    No model was called to produce this and no credentials were needed.
    """

    prompt_id: types_schema.PromptID  # str
    """The prompt that was rendered."""

    slug: str
    """Slug of the prompt that was rendered."""

    version_id: types_schema.PromptVersionID  # str
    """The exact version that was rendered."""

    label: Optional[str] = None
    """The label that was asked for, when the version came from one."""

    prompt_type: PromptType
    """Whether this holds text or messages."""

    text: Optional[str] = None
    """Rendered text. Set for text prompts only."""

    messages: Optional[List[Dict[str, Any]]] = None
    """Rendered provider-neutral message dictionaries. Chat prompts only."""

    model_args: Dict[str, Any] = pydantic.Field(default_factory=dict)
    """Persisted model defaults merged with the caller's overrides."""

    response_format: Optional[Dict[str, Any]] = None
    """Response-format metadata, kept out of `model_args` on purpose."""

    rendered_content_hash: str
    """Hash of the rendered content."""


class ResolvedPrompt(serial_utils.SerialModel):
    """One exact prompt version, plus strict local rendering."""

    prompt: Prompt
    """The prompt identity."""

    version: PromptVersion
    """The exact version that was resolved."""

    label: Optional[str] = None
    """The label that was requested, when a label was used."""

    @property
    def version_id(self) -> types_schema.PromptVersionID:
        """The exact version identifier."""

        return self.version.version_id

    def _validate_values(
        self, values: Dict[str, Any], strict: bool
    ) -> Dict[str, Any]:
        declared = set(self.version.variables)
        supplied = set(values)

        missing = sorted(declared - supplied)
        unexpected = sorted(supplied - declared)

        if missing:
            raise VariableError(f"Missing values for variables: {missing}.")
        if unexpected:
            if strict:
                raise VariableError(
                    f"Unexpected values for variables: {unexpected}."
                )
            return {name: values[name] for name in declared}

        return values

    def model_args(
        self, model_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Persisted model defaults merged with `model_overrides`.

        Neither the stored defaults nor the supplied overrides are mutated.

        Args:
            model_overrides: Caller settings that win over the defaults.

        Returns:
            A new dictionary.
        """

        merged = dict(self.version.model_defaults)
        if model_overrides:
            merged.update(model_overrides)
        return merged

    def render(
        self,
        model_overrides: Optional[Dict[str, Any]] = None,
        strict: bool = True,
        **values: Any,
    ) -> RenderedPrompt:
        """Render this version locally.

        Args:
            model_overrides: Model settings that win over the persisted
                defaults.
            strict: When true, values that no declared variable uses are an
                error rather than being dropped. Missing values are always an
                error.
            **values: One entry per declared variable.

        Returns:
            A [RenderedPrompt][trulens.core.schema.prompt.RenderedPrompt] with
            text or ordered message dictionaries, the merged model arguments,
            and the response format kept separate.

        Raises:
            VariableError: If the supplied values do not match the declared
                variables.
        """

        checked = self._validate_values(values, strict=strict)

        text: Optional[str] = None
        messages: Optional[List[Dict[str, Any]]] = None

        if self.version.prompt_type is PromptType.TEXT:
            text = _interpolate(self.version.text or "", checked)
            content = [text]
        else:
            messages = [
                {
                    "role": message.role.value,
                    "content": _interpolate_content(message.content, checked),
                }
                for message in self.version.messages or []
            ]
            content = [
                string
                for message in messages
                for string in _content_strings(message["content"])
            ]

        return RenderedPrompt(
            prompt_id=self.prompt.prompt_id,
            slug=self.prompt.slug,
            version_id=self.version.version_id,
            label=self.label,
            prompt_type=self.version.prompt_type,
            text=text,
            messages=messages,
            model_args=self.model_args(model_overrides),
            response_format=self.version.response_format,
            rendered_content_hash=json_utils.obj_id_of_obj(
                {
                    "prompt_type": self.version.prompt_type.value,
                    "content": content,
                },
                prefix="rendered",
            ),
        )

    def build(
        self,
        model_overrides: Optional[Dict[str, Any]] = None,
        strict: bool = True,
        **values: Any,
    ) -> List[Dict[str, Any]]:
        """Render a chat version straight to message dictionaries.

        Args:
            model_overrides: Model settings that win over the persisted
                defaults. Accepted so that the signature matches
                [render][trulens.core.schema.prompt.ResolvedPrompt.render].
            strict: See
                [render][trulens.core.schema.prompt.ResolvedPrompt.render].
            **values: One entry per declared variable.

        Returns:
            Ordered provider-neutral message dictionaries.

        Raises:
            ValueError: If this is a text prompt.
        """

        if self.version.prompt_type is not PromptType.CHAT:
            raise ValueError(
                "build() is for chat prompts. Use render() for text prompts."
            )

        rendered = self.render(
            model_overrides=model_overrides, strict=strict, **values
        )
        return rendered.messages or []


# HACK013: Need these if using __future__.annotations .
PromptMessage.model_rebuild()
Prompt.model_rebuild()
PromptVersion.model_rebuild()
PromptLabel.model_rebuild()
PromptLabelHistory.model_rebuild()
RenderedPrompt.model_rebuild()
ResolvedPrompt.model_rebuild()
