"""Type aliases."""

from typing import Dict
import uuid

import typing_extensions

RecordID: typing_extensions.TypeAlias = str
"""Unique identifier for a record.

By default these hashes of record content as json.
[Record.record_id][trulens_eval.schema.record.Record.record_id].
"""

CallID: typing_extensions.TypeAlias = str
"""Unique identifier for a record app call.

See [RecordAppCall.call_id][trulens_eval.schema.record.RecordAppCall.call_id].
"""


def new_call_id() -> CallID:
    """Generate a new call id."""
    return str(uuid.uuid4())


AppID: typing_extensions.TypeAlias = str
"""Unique identifier for an app.

By default these are hashes of app content as json.
See [App.app_id][trulens_eval.schema.app.App.app_id].
"""

Tags: typing_extensions.TypeAlias = str
"""Tags for an app or record.

See [App.tags][trulens_eval.schema.app.App.tags] and
[Record.tags][trulens_eval.schema.record.Record.tags].
"""

Metadata: typing_extensions.TypeAlias = Dict
"""Metadata for an app or record.

See [App.meta][trulens_eval.schema.app.App.meta] and
[Record.meta][trulens_eval.schema.record.Record.meta].
"""

FeedbackDefinitionID: typing_extensions.TypeAlias = str
"""Unique identifier for a feedback definition.

By default these are hashes of feedback definition content as json. See
[FeedbackDefinition.feedback_definition_id][trulens_eval.schema.feedback.FeedbackDefinition.feedback_definition_id].
"""

FeedbackResultID: typing_extensions.TypeAlias = str
"""Unique identifier for a feedback result.

By default these are hashes of feedback result content as json. See
[FeedbackResult.feedback_result_id][trulens_eval.schema.feedback.FeedbackResult.feedback_result_id].
"""
