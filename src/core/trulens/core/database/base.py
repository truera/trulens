from __future__ import annotations

import abc
from datetime import datetime
import json
import logging
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd
from trulens.core.schema import app as app_schema
from trulens.core.schema import dataset as dataset_schema
from trulens.core.schema import event as event_schema
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import groundtruth as groundtruth_schema
from trulens.core.schema import record as record_schema
from trulens.core.schema import types as types_schema
from trulens.core.schema.event import Event
from trulens.core.utils import json as json_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import text as text_utils
from trulens.otel.semconv.trace import BASE_SCOPE
from trulens.otel.semconv.trace import ResourceAttributes
from trulens.otel.semconv.trace import SpanAttributes

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


class BaseAppsExtractor:
    """Utilities for creating dataframes from orm instances."""

    app_cols = ["app_name", "app_version", "app_id", "app_json", "type"]
    rec_cols = [
        "record_id",
        "input",
        "output",
        "tags",
        "record_json",
        "cost_json",
        "perf_json",
        "ts",
    ]
    extra_cols = ["latency", "total_tokens", "total_cost", "num_events"]
    all_cols = app_cols + rec_cols + extra_cols


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
        app_versions: Optional[List[types_schema.AppVersion]] = None,
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
            app_versions: If given, retrieve only the records for the given app versions.
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

    def _get_records_and_feedback_otel_from_events(
        self,
        events: List[Event],
        app_ids: Optional[List[str]] = None,
        app_name: Optional[types_schema.AppName] = None,
    ) -> Tuple[pd.DataFrame, Sequence[str]]:
        if not events:
            # Return empty dataframe with expected columns
            logger.warning(
                f"No events found for app_name: {app_name}, app_ids: {app_ids}"
            )
            return pd.DataFrame(columns=BaseAppsExtractor.all_cols), []

        # Group events by record_id
        record_events = {}
        for event in events:
            record_attributes = self._get_event_record_attributes_otel(event)
            resource_attributes = self._get_event_resource_attributes_otel(
                event
            )
            record_id = record_attributes.get(SpanAttributes.RECORD_ID)
            if not record_id:
                continue
            app_name, app_version, app_id, _ = self.extract_app_and_run_info(
                record_attributes, resource_attributes
            )
            if app_ids and app_id not in app_ids:
                # TODO(otel):
                # This may screw up the pagination and can be slow due to it
                # looking at possibly a lot more events if there are many
                # that don't have app ids.
                # In the future we should either:
                # 1. Remove app ids if we're going to assume they're some
                #    complex function of app_name and app_version that's
                #    hard to replicate for non-TruLens users that want to
                #    still use our evaluation/feedback stuff.
                # 2. Have the app ids be from some source of truth like the
                #    app table but this doesn't work as easily for the
                #    Snowflake side.
                logger.info(f"Computed {app_id} not in {app_ids}!")
                continue

            if record_id not in record_events:
                record_events[record_id] = {
                    "events": [],
                    "app_name": app_name,
                    "app_version": app_version,
                    "app_id": app_id,
                    "input": "",  # Initialize to empty string, filled below
                    "output": "",  # Initialize to empty string, filled below
                    "tags": "",  # Not present in OTEL, use empty string
                    "ts": pd.NaT,  # Initialize to empty value, filled below
                    "latency": 0.0,  # Initialize to 0.0, filled below
                    "total_tokens": 0,  # Initialize to 0, calculated below
                    "total_cost": 0.0,  # Initialize to 0.0, calculated below
                    "cost_currency": "USD",  # Initialize to "USD", calculated below
                    "feedback_results": {},  # Initialize to empty map, calculated below
                }

            record_events[record_id]["events"].append(event)

            # Check if the span is of type RECORD_ROOT
            if (
                record_attributes.get(SpanAttributes.SPAN_TYPE)
                == SpanAttributes.SpanType.RECORD_ROOT.value
            ):
                record_events[record_id]["input_id"] = record_attributes.get(
                    SpanAttributes.INPUT_ID, ""
                )
                record_events[record_id]["input"] = record_attributes.get(
                    SpanAttributes.RECORD_ROOT.INPUT, ""
                )
                record_events[record_id]["output"] = record_attributes.get(
                    SpanAttributes.RECORD_ROOT.OUTPUT, ""
                )
                # NOTE: We grab timestamps from the RECORD_ROOT span because it provides a
                # more accurate duration/latency.
                record_events[record_id]["ts"] = event.start_timestamp
                record_events[record_id]["latency"] = (
                    event.timestamp - event.start_timestamp
                ).total_seconds()

            # Check if the span has cost info (tokens, cost, currency), and update record events
            self._update_cost_info_otel(
                record_events[record_id],
                record_attributes,
                include_tokens=True,
            )

        # Process feedback results
        feedback_col_names = []
        for record_id, record_data in record_events.items():
            for event in record_data["events"]:
                record_attributes = self._get_event_record_attributes_otel(
                    event
                )

                # Check if the span is of type EVAL or EVAL_ROOT
                if record_attributes.get(SpanAttributes.SPAN_TYPE) in [
                    SpanAttributes.SpanType.EVAL.value,
                    SpanAttributes.SpanType.EVAL_ROOT.value,
                ]:
                    metric_name = record_attributes.get(
                        SpanAttributes.EVAL.METRIC_NAME
                    )
                    if not metric_name:
                        logger.warning(
                            f"Skipping eval span for record_id: {record_id}, no metric name found"
                        )
                        continue

                    # Add feedback name to column names if not present
                    if metric_name not in feedback_col_names:
                        feedback_col_names.append(metric_name)

                    # Initialize feedback result if not present
                    if metric_name not in record_data["feedback_results"]:
                        record_data["feedback_results"][metric_name] = {
                            "mean_score": None,
                            "calls": [],
                            "total_cost": 0.0,
                            "cost_currency": "USD",  # Initialize to USD, calculated below
                            "direction": None,
                        }

                    # Update feedback result
                    # TODO(otel): This isn't going to work if there are multiple with the same name.
                    feedback_result = record_data["feedback_results"][
                        metric_name
                    ]

                    eval_root_score = record_attributes.get(
                        SpanAttributes.EVAL_ROOT.SCORE, None
                    )

                    if (
                        record_attributes.get(SpanAttributes.SPAN_TYPE)
                        == SpanAttributes.SpanType.EVAL_ROOT.value
                    ):
                        # NOTE: EVAL_ROOT.SCORE should provide the mean score of all related EVAL spans
                        feedback_result["mean_score"] = eval_root_score
                        # TODO(SNOW-2112879): HIGHER_IS_BETTER has not been populated in the OTEL spans yet
                        feedback_result["direction"] = record_attributes.get(
                            SpanAttributes.EVAL_ROOT.HIGHER_IS_BETTER,
                            None,
                        )
                        # Add call data for EVAL_ROOT spans
                        args_span_id = self._extract_namespaced_attributes(
                            record_attributes,
                            SpanAttributes.EVAL_ROOT.ARGS_SPAN_ID,
                        )
                        args_span_attribute = (
                            self._extract_namespaced_attributes(
                                record_attributes,
                                SpanAttributes.EVAL_ROOT.ARGS_SPAN_ATTRIBUTE,
                            )
                        )

                        call_data = {
                            "span_type": record_attributes.get(
                                SpanAttributes.SPAN_TYPE
                            ),
                            "eval_root_id": record_attributes.get(
                                SpanAttributes.EVAL.EVAL_ROOT_ID
                            ),
                            "timestamp": record_data["ts"],
                            "args_span_id": args_span_id,
                            "args_span_attribute": args_span_attribute,
                        }
                        feedback_result["calls"].append(call_data)

                        # Update feedback result with cost info if available
                        self._update_cost_info_otel(
                            feedback_result, record_attributes
                        )

                    if (
                        record_attributes.get(SpanAttributes.SPAN_TYPE)
                        == SpanAttributes.SpanType.EVAL.value
                    ):
                        # Extract namespaced attributes using the helper method
                        kwargs = self._extract_namespaced_attributes(
                            record_attributes, SpanAttributes.CALL.KWARGS
                        )

                        call_data = {
                            "span_type": record_attributes.get(
                                SpanAttributes.SPAN_TYPE
                            ),
                            "args": kwargs,
                            "ret": record_attributes.get(
                                SpanAttributes.EVAL.SCORE
                            ),
                            "eval_root_id": record_attributes.get(
                                SpanAttributes.EVAL.EVAL_ROOT_ID
                            ),
                            "timestamp": record_data["ts"],
                            "meta": {
                                "explanation": record_attributes.get(
                                    SpanAttributes.EVAL.EXPLANATION
                                ),
                                "metadata": record_attributes.get(
                                    SpanAttributes.EVAL.METADATA, {}
                                ),
                            },
                        }
                        feedback_result["calls"].append(call_data)

                        # Update feedback result with cost info if available
                        self._update_cost_info_otel(
                            feedback_result, record_attributes
                        )

        # Create dataframe
        records_data = []
        for record_id, record_data in record_events.items():
            # TODO: audit created jsons for correctness (app_json, record_json, cost_json, perf_json)

            app_json = {
                "app_name": record_data["app_name"],
                "app_version": record_data["app_version"],
                "app_id": record_data["app_id"],
            }

            record_json = {
                "record_id": record_id,
                "app_id": record_data["app_id"],
                "input": record_data["input"],
                "output": record_data["output"],
                "tags": record_data["tags"],
                "ts": record_data["ts"],
                "meta": {},
            }

            cost_json = {
                "n_tokens": record_data["total_tokens"],
                # TODO: convert to map (see comment: https://github.com/truera/trulens/pull/1939#discussion_r2054802093)
                "cost": record_data["total_cost"],
            }

            perf_json = {
                "start_time": record_data["ts"],
                "end_time": record_data["ts"]
                + pd.Timedelta(seconds=record_data["latency"]),
            }

            # Create record row
            record_row = {
                "app_id": record_data["app_id"],
                "app_name": record_data["app_name"],
                "app_version": record_data["app_version"],
                "app_json": app_json,
                # TODO(nit): consider using a constant here
                "type": "SPAN",  # Default type as per orm.py
                "record_id": record_id,
                "input_id": record_data.get("input_id"),
                "input": record_data["input"],
                "output": record_data["output"],
                "tags": record_data["tags"],
                "record_json": record_json,
                "cost_json": cost_json,
                "perf_json": perf_json,
                "ts": record_data["ts"],
                "latency": record_data["latency"],
                "total_tokens": record_data["total_tokens"],
                # TODO: convert to map (see comment: https://github.com/truera/trulens/pull/1939#discussion_r2054802093)
                "total_cost": record_data["total_cost"],
                "cost_currency": record_data["cost_currency"],
                "num_events": len(record_data["events"]),
            }

            # Add feedback results
            for feedback_name, feedback_result in record_data[
                "feedback_results"
            ].items():
                # NOTE: we use the mean score as the feedback result
                record_row[feedback_name] = feedback_result["mean_score"]

                record_row[f"{feedback_name}_calls"] = feedback_result["calls"]
                record_row[
                    f"{feedback_name} feedback cost in {feedback_result['cost_currency']}"
                ] = feedback_result["total_cost"]
                record_row[f"{feedback_name} direction"] = feedback_result[
                    "direction"
                ]

            records_data.append(record_row)

        # Create dataframe
        df = pd.DataFrame(records_data)

        # Ensure that all expected columns are present
        for col in BaseAppsExtractor.all_cols:
            if col not in df.columns:
                logger.warning(
                    f"Column {col} not found in dataframe, setting to None."
                )
                df[col] = None

        return df, feedback_col_names

    def _get_event_record_attributes_otel(self, event: Event) -> Dict[str, Any]:
        """Get the record attributes from the event.

        This implementation differs from the pre-OTEL implementation by using the
        `record_attributes` field of the event.

        Args:
            event: The event to extract the record attributes from.

        Returns:
            Dict[str, Any]: The record attributes from the event.
        """
        record_attributes = event.record_attributes
        if not isinstance(record_attributes, dict):
            try:
                record_attributes = json.loads(record_attributes)
            except (json.JSONDecodeError, TypeError):
                logger.error(
                    f"Failed to decode record attributes as JSON: {record_attributes}",
                )

        return record_attributes

    def _get_event_resource_attributes_otel(
        self, event: Event
    ) -> Dict[str, Any]:
        """Get the resource attributes from the event.

        This implementation differs from the pre-OTEL implementation by using the
        `resource_attributes` field of the event.

        Args:
            event: The event to extract the resource attributes from.

        Returns:
            Dict[str, Any]: The resource attributes from the event.
        """
        resource_attributes = event.resource_attributes
        if not isinstance(resource_attributes, dict):
            try:
                resource_attributes = json.loads(resource_attributes)
            except (json.JSONDecodeError, TypeError):
                logger.error(
                    f"Failed to decode resource attributes as JSON: {resource_attributes}",
                )

        return resource_attributes

    def _update_cost_info_otel(
        self,
        target_dict: dict,
        record_attributes: dict,
        include_tokens: bool = False,
    ):
        """Update cost information in the target dictionary.

        Args:
            target_dict: Dictionary to update with cost information
            record_attributes: Source attributes containing cost information
            include_tokens: Whether to update token count (only for record_events)
        """
        if any(
            key.startswith(SpanAttributes.COST.base)
            for key in record_attributes
        ):
            if include_tokens:
                target_dict["total_tokens"] += record_attributes.get(
                    SpanAttributes.COST.NUM_TOKENS, 0
                )

            target_dict["total_cost"] += record_attributes.get(
                SpanAttributes.COST.COST, 0.0
            )
            target_dict["cost_currency"] = record_attributes.get(
                SpanAttributes.COST.CURRENCY, "USD"
            )

        # TODO(SNOW-2061174): convert to map (see comment: https://github.com/truera/trulens/pull/1939#discussion_r2054802093)
        # Add to total_cost map
        # cost = record_attributes.get(SpanAttributes.COST.COST, 0.0)
        # currency = record_attributes.get(SpanAttributes.COST.CURRENCY, "USD")
        # if currency not in record_events[record_id]["total_cost"]:
        #     record_events[record_id]["total_cost"][currency] = 0.0
        # record_events[record_id]["total_cost"][currency] += cost

    @staticmethod
    def extract_app_and_run_info(
        attributes: Dict[str, Any], resource_attributes: Dict[str, Any]
    ) -> Tuple[str, str, str, str]:
        """Get app info from attributes.

        Args:
            attributes: Span attributes of record root.
            resource_attributes: Resource attributes of record root.

        Returns:
            Tuple of: app name, app version, app id, and run name.
        """

        def get_value(keys: List[str]) -> Optional[str]:
            for key in keys:
                for attr in [resource_attributes, attributes]:
                    if key in attr:
                        return attr[key]
            return None

        app_name = get_value([
            f"snow.{BASE_SCOPE}.object.name",
            ResourceAttributes.APP_NAME,
        ])
        app_version = get_value([
            f"snow.{BASE_SCOPE}.object.version.name",
            ResourceAttributes.APP_VERSION,
        ])
        app_id = get_value([ResourceAttributes.APP_ID])
        if app_id is None:
            app_id = app_schema.AppDefinition._compute_app_id(
                app_name, app_version
            )
        run_name = get_value([
            f"snow.{BASE_SCOPE}.run.name",
            SpanAttributes.RUN_NAME,
        ])
        return app_name, app_version, app_id, run_name
