"""Type aliases."""

from typing import Dict
import uuid

from trulens.core._utils.pycompat import TypeAlias  # import style exception

RecordID: TypeAlias = str
"""Unique identifier for a record.

By default these hashes of record content as json.
[Record.record_id][trulens.core.schema.record.Record.record_id].
"""

ConversationID: TypeAlias = str
"""Unique identifier for a conversation."""

CallID: TypeAlias = str
"""Unique identifier for a record app call.

See [RecordAppCall.call_id][trulens.core.schema.record.RecordAppCall.call_id].
"""


def new_call_id() -> CallID:
    """Generate a new call id."""
    return str(uuid.uuid4())


AppID: TypeAlias = str
"""Unique identifier for an app.

By default these are hashes of app content as json.
See [AppDefinition.app_id][trulens.core.schema.app.AppDefinition.app_id].
"""

AppName: TypeAlias = str
"""Unique App name.

See [AppDefinition.app_name][trulens.core.schema.app.AppDefinition.app_name].
"""

AppVersion: TypeAlias = str
"""Version identifier for an app.

See [AppDefinition.app_version][trulens.core.schema.app.AppDefinition.app_version].
"""

RunName: TypeAlias = str
"""Run name."""

Tags: TypeAlias = str
"""Tags for an app or record.

See [AppDefinition.tags][trulens.core.schema.app.AppDefinition.tags] and
[Record.tags][trulens.core.schema.record.Record.tags].
"""

Metadata: TypeAlias = Dict
"""Metadata for an app, record, groundtruth, or dataset.

See [AppDefinition.metadata][trulens.core.schema.app.AppDefinition.metadata],
[Record.meta][trulens.core.schema.record.Record.meta], [GroundTruth.meta][trulens.core.schema.groundtruth.GroundTruth.meta], and
[Dataset.meta][trulens.core.schema.dataset.Dataset.meta].
"""

FeedbackDefinitionID: TypeAlias = str
"""Unique identifier for a feedback definition.

By default these are hashes of feedback definition content as json. See
[FeedbackDefinition.feedback_definition_id][trulens.core.schema.feedback.FeedbackDefinition.feedback_definition_id].
"""

FeedbackResultID: TypeAlias = str
"""Unique identifier for a feedback result.

By default these are hashes of feedback result content as json. See
[FeedbackResult.feedback_result_id][trulens.core.schema.feedback.FeedbackResult].
"""

GroundTruthID: TypeAlias = str
"""Unique identifier for a groundtruth.

By default these are hashes of ground truth content as json.
"""

DatasetID: TypeAlias = str
"""Unique identifier for a dataset.

By default these are hashes of dataset content as json.
"""

DatasetVersionID: TypeAlias = str
"""Unique identifier for an immutable dataset version.

These are content-addressed: the id is a hash of the ordered contents of the
version, so publishing identical content twice yields the same id. See
[DatasetVersion.dataset_version_id][trulens.core.schema.dataset.DatasetVersion.dataset_version_id].
"""

DatasetVersionItemID: TypeAlias = str
"""Unique identifier for a single example within a dataset version.

These are content-addressed from the normalized example content and the
optional caller-supplied input id, so the same example carries the same id
across versions. See
[DatasetVersionItem.item_id][trulens.core.schema.dataset.DatasetVersionItem.item_id].
"""

EventID: TypeAlias = str
"""Unique identifier for a event.
"""
