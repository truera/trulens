"""Prompt-management operations over the configured TruLens database.

These functions back the `TruSession` prompt methods. They read and write
prompt definitions, versions, and labels; they never render with a model and
never need credentials.

Label lookups always resolve from the database. The process-local
[LabelCache][trulens.core.prompt.LabelCache] only skips repeated reads within
its time-to-live, so correctness never depends on it.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from trulens.core.database import base as core_db
from trulens.core.schema import prompt as prompt_schema
from trulens.core.schema import types as types_schema

logger = logging.getLogger(__name__)

DEFAULT_LABEL_CACHE_TTL: float = 0.0
"""Default time-to-live in seconds. Zero means every lookup hits the database."""


class LabelCache:
    """A process-local time-to-live cache of label to exact version.

    This is an optimisation only. Entries expire after `ttl` seconds and any
    caller can bypass or clear it.
    """

    def __init__(self, ttl: float = DEFAULT_LABEL_CACHE_TTL):
        self.ttl = ttl
        self._entries: Dict[
            Tuple[types_schema.PromptID, str],
            Tuple[types_schema.PromptVersionID, float],
        ] = {}
        self._lock = threading.Lock()

    def get(
        self, prompt_id: types_schema.PromptID, label: str
    ) -> Optional[types_schema.PromptVersionID]:
        """Get a cached version id, or None when absent or expired."""

        if self.ttl <= 0:
            return None

        with self._lock:
            entry = self._entries.get((prompt_id, label))
            if entry is None:
                return None
            version_id, stored_at = entry
            if time.monotonic() - stored_at >= self.ttl:
                del self._entries[(prompt_id, label)]
                return None
            return version_id

    def set(
        self,
        prompt_id: types_schema.PromptID,
        label: str,
        version_id: types_schema.PromptVersionID,
    ) -> None:
        """Store a version id for a label."""

        if self.ttl <= 0:
            return

        with self._lock:
            self._entries[(prompt_id, label)] = (version_id, time.monotonic())

    def invalidate(
        self,
        prompt_id: Optional[types_schema.PromptID] = None,
        label: Optional[str] = None,
    ) -> None:
        """Drop one entry, every entry of one prompt, or the whole cache."""

        with self._lock:
            if prompt_id is None:
                self._entries.clear()
            elif label is None:
                for key in [
                    key for key in self._entries if key[0] == prompt_id
                ]:
                    del self._entries[key]
            else:
                self._entries.pop((prompt_id, label), None)


def create_prompt(
    db: core_db.DB,
    slug: str,
    name: Optional[str] = None,
    prompt_type: Union[
        prompt_schema.PromptType, str
    ] = prompt_schema.PromptType.TEXT,
    description: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
) -> prompt_schema.Prompt:
    """Create a prompt, or update the metadata of an existing slug.

    Args:
        db: The database to write to.
        slug: Stable key such as `support-assistant`.
        name: Display name. Defaults to the slug.
        prompt_type: `text` or `chat`. Fixed after creation.
        description: Free-text description.
        tags: Tags for grouping.

    Returns:
        The stored prompt.
    """

    prompt = prompt_schema.Prompt(
        slug=slug,
        name=name,
        prompt_type=prompt_type,
        description=description,
        tags=list(tags) if tags else [],
    )
    db.insert_prompt(prompt)
    return prompt


def create_prompt_version(
    db: core_db.DB,
    prompt: Union[prompt_schema.Prompt, types_schema.PromptID, str],
    text: Optional[str] = None,
    messages: Optional[
        Sequence[Union[prompt_schema.PromptMessage, Dict[str, Any]]]
    ] = None,
    variables: Optional[Sequence[str]] = None,
    model_defaults: Optional[Dict[str, Any]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    change_note: Optional[str] = None,
    parent_version_id: Optional[types_schema.PromptVersionID] = None,
    created_by: Optional[str] = None,
    cache: Optional[LabelCache] = None,
) -> prompt_schema.PromptVersion:
    """Create an immutable version and move `latest` onto it.

    Creating the same content twice returns the same version.

    Args:
        db: The database to write to.
        prompt: A prompt, prompt id, or slug.
        text: Template string for a text prompt.
        messages: Ordered messages for a chat prompt.
        variables: Declared variable names. Inferred from the content when
            omitted.
        model_defaults: Provider-neutral model settings.
        response_format: Response-format metadata for provider adapters.
        change_note: Why the version was created.
        parent_version_id: The version this one derives from. Defaults to
            whatever `latest` currently points at.
        created_by: Who created the version.
        cache: Cache to invalidate for the moved `latest` label.

    Returns:
        The stored version.
    """

    resolved_prompt = as_prompt(db, prompt)

    if parent_version_id is None:
        current_latest = db.get_prompt_label(
            resolved_prompt.prompt_id, prompt_schema.LATEST_LABEL
        )
        if current_latest is not None:
            parent_version_id = current_latest.version_id

    version = prompt_schema.PromptVersion(
        prompt_id=resolved_prompt.prompt_id,
        prompt_type=resolved_prompt.prompt_type,
        text=text,
        messages=messages,
        variables=variables,
        model_defaults=model_defaults or {},
        response_format=response_format,
        change_note=change_note,
        parent_version_id=parent_version_id,
        created_by=created_by,
    )
    db.insert_prompt_version(version)

    if cache is not None:
        cache.invalidate(resolved_prompt.prompt_id, prompt_schema.LATEST_LABEL)

    return version


def set_prompt_label(
    db: core_db.DB,
    prompt: Union[prompt_schema.Prompt, types_schema.PromptID, str],
    label: str,
    version: Union[prompt_schema.PromptVersion, types_schema.PromptVersionID],
    moved_by: Optional[str] = None,
    cache: Optional[LabelCache] = None,
) -> prompt_schema.PromptLabel:
    """Point a label at one exact version.

    Rolling back is the same call with an older version.

    Args:
        db: The database to write to.
        prompt: A prompt, prompt id, or slug.
        label: The label name, for example `production`.
        version: The version or version id to point at.
        moved_by: Caller label written to the history entry.
        cache: Cache to invalidate for this label.

    Returns:
        The resulting label pointer.
    """

    resolved_prompt = as_prompt(db, prompt)
    version_id = (
        version.version_id
        if isinstance(version, prompt_schema.PromptVersion)
        else version
    )

    pointer = db.set_prompt_label(
        prompt_id=resolved_prompt.prompt_id,
        label=label,
        version_id=version_id,
        moved_by=moved_by,
    )

    if cache is not None:
        cache.invalidate(resolved_prompt.prompt_id, label)

    return pointer


def resolve_prompt(
    db: core_db.DB,
    prompt: Union[prompt_schema.Prompt, types_schema.PromptID, str],
    label: Optional[str] = None,
    version_id: Optional[types_schema.PromptVersionID] = None,
    cache: Optional[LabelCache] = None,
    use_cache: bool = True,
) -> prompt_schema.ResolvedPrompt:
    """Resolve a prompt to one exact version.

    Args:
        db: The database to read from.
        prompt: A prompt, prompt id, or slug.
        label: Label to resolve. Defaults to `latest` when no `version_id` is
            given.
        version_id: Exact version to load. Wins over `label`.
        cache: Optional label cache.
        use_cache: Set false to bypass `cache` for this call.

    Returns:
        A [ResolvedPrompt][trulens.core.schema.prompt.ResolvedPrompt].

    Raises:
        ValueError: If the prompt, label, or version does not exist.
    """

    resolved_prompt = as_prompt(db, prompt)
    requested_label = None

    if version_id is None:
        requested_label = label or prompt_schema.LATEST_LABEL

        cached = (
            cache.get(resolved_prompt.prompt_id, requested_label)
            if cache is not None and use_cache
            else None
        )
        if cached is not None:
            version_id = cached
        else:
            pointer = db.get_prompt_label(
                resolved_prompt.prompt_id, requested_label
            )
            if pointer is None:
                raise ValueError(
                    f"Prompt {resolved_prompt.slug!r} has no label "
                    f"{requested_label!r}."
                )
            version_id = pointer.version_id
            if cache is not None:
                cache.set(
                    resolved_prompt.prompt_id, requested_label, version_id
                )

    version = db.get_prompt_version(version_id)
    if version is None:
        raise ValueError(f"No prompt version with id {version_id!r} exists.")

    return prompt_schema.ResolvedPrompt(
        prompt=resolved_prompt,
        version=version,
        label=requested_label,
    )


def as_prompt(
    db: core_db.DB,
    prompt: Union[prompt_schema.Prompt, types_schema.PromptID, str],
) -> prompt_schema.Prompt:
    """Accept a prompt, a prompt id, or a slug and return the stored prompt.

    Args:
        db: The database to read from.
        prompt: A prompt, prompt id, or slug.

    Returns:
        The stored prompt.

    Raises:
        ValueError: If no such prompt exists.
    """

    if isinstance(prompt, prompt_schema.Prompt):
        return prompt

    found = db.get_prompt(prompt_id=prompt) or db.get_prompt(slug=prompt)
    if found is None:
        raise ValueError(f"No prompt with id or slug {prompt!r} exists.")
    return found
