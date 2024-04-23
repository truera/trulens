"""Type aliases."""

from typing import Dict
import typing_extensions

RecordID: typing_extensions.TypeAlias = str
"""Unique identifier for a record."""

AppID: typing_extensions.TypeAlias = str
"""Unique identifier for an app."""

Tags: typing_extensions.TypeAlias = str
"""Tags for an app or record."""

Metadata: typing_extensions.TypeAlias = Dict
"""Metadata for an app or record."""

FeedbackDefinitionID: typing_extensions.TypeAlias = str
"""Unique identifier for a feedback definition."""

FeedbackResultID: typing_extensions.TypeAlias = str
"""Unique identifier for a feedback result."""
