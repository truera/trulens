"""Type aliases."""

from typing import Dict
import uuid

import typing_extensions

RecordID: typing_extensions.TypeAlias = str
"""Unique identifier for a record.

By default these hashes of record content as json.
[Record.record_id][trulens.core.schema.record.Record.record_id].
"""

CallID: typing_extensions.TypeAlias = str
"""Unique identifier for a record app call.

See [RecordAppCall.call_id][trulens.core.schema.record.RecordAppCall.call_id].
"""


def new_call_id() -> CallID:
    """Generate a new call id."""
    return str(uuid.uuid4())


AppID: typing_extensions.TypeAlias = str
"""Unique identifier for an app.

By default these are hashes of app content as json.
See [AppDefinition.app_id][trulens.core.schema.app.AppDefinition.app_id].
"""

AppName: typing_extensions.TypeAlias = str
"""Unique App name.

See [AppDefinition.app_name][trulens.core.schema.app.AppDefinition.app_name].
"""

AppVersion: typing_extensions.TypeAlias = str
"""Version identifier for an app.

See [AppDefinition.app_version][trulens.core.schema.app.AppDefinition.app_version].
"""

Tags: typing_extensions.TypeAlias = str
"""Tags for an app or record.

See [AppDefinition.tags][trulens.core.schema.app.AppDefinition.tags] and
[Record.tags][trulens.core.schema.record.Record.tags].
"""

Metadata: typing_extensions.TypeAlias = Dict
"""Metadata for an app, record, groundtruth, or dataset.

See [AppDefinition.metadata][trulens.core.schema.app.AppDefinition.metadata],
[Record.meta][trulens.core.schema.record.Record.meta], [Groundtruth.metadata][trulens.core.schema.groundtruth.Groundtruth.metadata], and
[Dataset.metadata][trulens.core.schema.dataset.Dataset.metadata].
"""

FeedbackDefinitionID: typing_extensions.TypeAlias = str
"""Unique identifier for a feedback definition.

By default these are hashes of feedback definition content as json. See
[FeedbackDefinition.feedback_definition_id][trulens.core.schema.feedback.FeedbackDefinition.feedback_definition_id].
"""

FeedbackResultID: typing_extensions.TypeAlias = str
"""Unique identifier for a feedback result.

By default these are hashes of feedback result content as json. See
[FeedbackResult.feedback_result_id][trulens.core.schema.feedback.FeedbackResult].
"""

GroundTruthID: typing_extensions.TypeAlias = str
"""Unique identifier for a groundtruth.

By default these are hashes of ground truth content as json.

See [Groundtruth.ground_truth_id][trulens.core.schema.groundtruth.Groundtruth.ground_truth_id].
"""

DatasetID: typing_extensions.TypeAlias = str
"""Unique identifier for a dataset.

By default these are hashes of dataset content as json.
See [Dataset.dataset_id][trulens.core.schema.dataset.Dataset.dataset_id].
"""
