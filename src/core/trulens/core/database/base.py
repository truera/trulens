import abc
from datetime import datetime
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
from trulens.core.schema import app as app_schema
from trulens.core.schema import dataset as dataset_schema
from trulens.core.schema import event as event_schema
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import groundtruth as groundtruth_schema
from trulens.core.schema import record as record_schema
from trulens.core.schema import types as types_schema
from trulens.core.utils import json as json_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import text as text_utils

NoneType = type(None)

logger = logging.getLogger(__name__)

MULTI_CALL_NAME_DELIMITER = ":::"

DEFAULT_DATABASE_PREFIX: str = "trulens_"
"""Default prefix for table names for trulens to use.

This includes alembic's version table.
"""

DEFAULT_DATABASE_FILE: str = "default.sqlite"
"""Filename for default sqlite database.

The sqlalchemy url for this default local sqlite database is `sqlite:///default.sqlite`.
"""

DEFAULT_DATABASE_REDACT_KEYS: bool = False
"""Default value for option to redact secrets before writing out data to database."""


class DB(serial_utils.SerialModel, abc.ABC, text_utils.WithIdentString):
    """Abstract definition of databases used by trulens.

    [SQLAlchemyDB][trulens.core.database.sqlalchemy.SQLAlchemyDB] is the main
    and default implementation of this interface.
    """

    redact_keys: bool = DEFAULT_DATABASE_REDACT_KEYS
    """Redact secrets before writing out data."""

    table_prefix: str = DEFAULT_DATABASE_PREFIX
    """Prefix for table names for trulens to use.

    May be useful in some databases where trulens is not the only app.
    """

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    # WithIdentString requirement:
    def _ident_str(self) -> str:
        return f"{self.__class__.__name__}"

    def _json_str_of_obj(self, obj: Any) -> str:
        return json_utils.json_str_of_obj(obj, redact_keys=self.redact_keys)

    @abc.abstractmethod
    def reset_database(self):
        """Delete all data."""

        raise NotImplementedError()

    @abc.abstractmethod
    def migrate_database(self, prior_prefix: Optional[str] = None):
        """Migrate the stored data to the current configuration of the database.

        Args:
            prior_prefix: If given, the database is assumed to have been
                reconfigured from a database with the given prefix. If not
                given, it may be guessed if there is only one table in the
                database with the suffix `alembic_version`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def check_db_revision(self):
        """Check that the database is up to date with the current trulens
        version.

        Raises:
            ValueError: If the database is not up to date.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_db_revision(self) -> Optional[str]:
        """Get the current revision of the database.

        Returns:
            The current revision of the database.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def insert_record(
        self,
        record: record_schema.Record,
    ) -> types_schema.RecordID:
        """
        Upsert a `record` into the database.

        Args:
            record: The record to insert or update.

        Returns:
            The id of the given record.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def batch_insert_record(
        self, records: List[record_schema.Record]
    ) -> List[types_schema.RecordID]:
        """
        Upsert a batch of records into the database.

        Args:
            records: The records to insert or update.

        Returns:
            The ids of the given records.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def insert_app(self, app: app_schema.AppDefinition) -> types_schema.AppID:
        """
        Upsert an `app` into the database.

        Args:
            app: The app to insert or update. Note that only the
                [AppDefinition][trulens.core.schema.app.AppDefinition] parts are serialized
                hence the type hint.

        Returns:
            The id of the given app.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def delete_app(self, app_id: types_schema.AppID) -> None:
        """
        Delete an `app` from the database.

        Args:
            app_id: The id of the app to delete.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_feedback_definition(
        self, feedback_definition: feedback_schema.FeedbackDefinition
    ) -> types_schema.FeedbackDefinitionID:
        """
        Upsert a `feedback_definition` into the database.

        Args:
            feedback_definition: The feedback definition to insert or update.
                Note that only the
                [FeedbackDefinition][trulens.core.schema.feedback.FeedbackDefinition]
                parts are serialized hence the type hint.

        Returns:
            The id of the given feedback definition.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def get_feedback_defs(
        self,
        feedback_definition_id: Optional[
            types_schema.FeedbackDefinitionID
        ] = None,
    ) -> pd.DataFrame:
        """Retrieve feedback definitions from the database.

        Args:
            feedback_definition_id: if provided, only the
                feedback definition with the given id is returned. Otherwise,
                all feedback definitions are returned.

        Returns:
            A dataframe with the feedback definitions.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_feedback(
        self,
        feedback_result: feedback_schema.FeedbackResult,
    ) -> types_schema.FeedbackResultID:
        """Upsert a `feedback_result` into the the database.

        Args:
            feedback_result: The feedback result to insert or update.

        Returns:
            The id of the given feedback result.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def batch_insert_feedback(
        self, feedback_results: List[feedback_schema.FeedbackResult]
    ) -> List[types_schema.FeedbackResultID]:
        """Upsert a batch of feedback results into the database.

        Args:
            feedback_results: The feedback results to insert or update.

        Returns:
            The ids of the given feedback results.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def get_feedback(
        self,
        record_id: Optional[types_schema.RecordID] = None,
        feedback_result_id: Optional[types_schema.FeedbackResultID] = None,
        feedback_definition_id: Optional[
            types_schema.FeedbackDefinitionID
        ] = None,
        status: Optional[
            Union[
                feedback_schema.FeedbackResultStatus,
                Sequence[feedback_schema.FeedbackResultStatus],
            ]
        ] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        shuffle: Optional[bool] = None,
        run_location: Optional[feedback_schema.FeedbackRunLocation] = None,
    ) -> pd.DataFrame:
        """Get feedback results matching a set of optional criteria:

        Args:
            record_id: Get only the feedback for the given record id.

            feedback_result_id: Get only the feedback for the given feedback
                result id.

            feedback_definition_id: Get only the feedback for the given feedback
                definition id.

            status: Get only the feedback with the given status. If a sequence
                of statuses is given, all feedback with any of the given
                statuses are returned.

            last_ts_before: get only results with `last_ts` before the
                given datetime.

            offset: index of the first row to return.

            limit: limit the number of rows returned.

            shuffle: shuffle the rows before returning them.

            run_location: Only get feedback functions with this run_location.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def get_feedback_count_by_status(
        self,
        record_id: Optional[types_schema.RecordID] = None,
        feedback_result_id: Optional[types_schema.FeedbackResultID] = None,
        feedback_definition_id: Optional[
            types_schema.FeedbackDefinitionID
        ] = None,
        status: Optional[
            Union[
                feedback_schema.FeedbackResultStatus,
                Sequence[feedback_schema.FeedbackResultStatus],
            ]
        ] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        shuffle: bool = False,
        run_location: Optional[feedback_schema.FeedbackRunLocation] = None,
    ) -> Dict[feedback_schema.FeedbackResultStatus, int]:
        """Get count of feedback results matching a set of optional criteria grouped by
        their status.

        See [get_feedback][trulens.core.database.base.DB.get_feedback] for the meaning of
        the the arguments.

        Returns:
            A mapping of status to the count of feedback results of that status
                that match the given filters.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def get_app(
        self, app_id: types_schema.AppID
    ) -> Optional[serial_utils.JSONized]:
        """Get the app with the given id from the database.

        Returns:
            The jsonized version of the app with the given id. Deserialization
                can be done with
                [App.model_validate][trulens.core.app.App.model_validate].

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_apps(
        self, app_name: Optional[types_schema.AppName] = None
    ) -> Iterable[serial_utils.JSONized[app_schema.AppDefinition]]:
        """Get all apps."""

        raise NotImplementedError()

    def update_app_metadata(
        self, app_id: types_schema.AppID, metadata: Dict[str, Any]
    ) -> Optional[app_schema.AppDefinition]:
        """Update the metadata of an app."""

    @abc.abstractmethod
    def get_records_and_feedback(
        self,
        app_ids: Optional[List[types_schema.AppID]] = None,
        app_name: Optional[types_schema.AppName] = None,
        app_version: Optional[types_schema.AppVersion] = None,
        record_ids: Optional[List[types_schema.RecordID]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Sequence[str]]:
        """Get records from the database.

        Args:
            app_ids: If given, retrieve only the records for the given apps.
                Otherwise all apps are retrieved.
            app_name: If given, retrieve only the records for the given app name.
            app_version: If given, retrieve only the records for the given app version.
            record_ids: Optional list of record IDs to filter by. Defaults to None.
            offset: Database row offset.
            limit: Limit on rows (records) returned.

        Returns:
            A DataFrame with the records.

            A list of column names that contain feedback results.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def insert_ground_truth(
        self, ground_truth: groundtruth_schema.GroundTruth
    ) -> types_schema.GroundTruthID:
        """Insert a ground truth entry into the database. The ground truth id is generated
        based on the ground truth content, so re-inserting is idempotent.

        Args:
            ground_truth: The ground truth entry to insert.

        Returns:
            The id of the given ground truth entry.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def batch_insert_ground_truth(
        self, ground_truths: List[groundtruth_schema.GroundTruth]
    ) -> List[types_schema.GroundTruthID]:
        """Insert a batch of ground truth entries into the database.

        Args:
            ground_truths: The ground truth entries to insert.

        Returns:
            The ids of the given ground truth entries.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_ground_truth(
        self,
        ground_truth_id: Optional[types_schema.GroundTruthID] = None,
    ) -> Optional[serial_utils.JSONized]:
        """Get the ground truth with the given id from the database."""

        raise NotImplementedError()

    @abc.abstractmethod
    def get_ground_truths_by_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Get all ground truths from the database from a particular dataset's name.

        Returns:
            A dataframe with the ground truths.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_virtual_ground_truth(
        self,
        user_table_name: str,
        user_schema_mapping: Dict[str, str],
        user_schema_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get all virtual ground truths from the database from a particular user table's name.

        Returns:
            A dataframe with the virtual ground truths.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def insert_dataset(
        self, dataset: dataset_schema.Dataset
    ) -> types_schema.DatasetID:
        """Insert a dataset into the database. The dataset id is generated based on the
        dataset content, so re-inserting is idempotent.

        Args:
            dataset: The dataset to insert.

        Returns:
            The id of the given dataset.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_datasets(self) -> pd.DataFrame:
        """Get all datasets from the database.

        Returns:
            A dataframe with the datasets.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def insert_event(self, event: event_schema.Event) -> types_schema.EventID:
        """Insert an event into the database.

        Args:
            event: The event to insert.

        Returns:
            The id of the given event.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_events(
        self,
        app_name: Optional[str],
        app_version: Optional[str],
        record_ids: Optional[List[str]],
        start_time: Optional[datetime],
    ) -> pd.DataFrame:
        """
        Get events from the database.

        Args:
            app_name: The app name to filter events by.
            app_version: The app version to filter events by.
            record_ids: The record ids to filter events by.
            start_time: The minimum time to consider events from.

        Returns:
            A pandas DataFrame of all relevant events.
        """
        raise NotImplementedError()
