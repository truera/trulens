import json
import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from snowflake.snowpark import AsyncJob
from snowflake.snowpark import Session
from snowflake.snowpark.row import Row
from trulens.connectors.snowflake.dao.enums import SourceType
from trulens.connectors.snowflake.dao.sql_utils import execute_query
from trulens.core.run import SUPPORTED_ENTRY_TYPES
from trulens.core.run import SupportedEntryType
from trulens.core.utils import json as json_utils

logger = logging.getLogger(__name__)


AIML_RUN_OPS_SYS_FUNC_TEMPLATE = (
    "SELECT SYSTEM$AIML_RUN_OPERATION('{method}', ?);"
)
METHOD_CREATE = "CREATE"
METHOD_GET = "GET"
METHOD_UPDATE = "UPDATE"
METHOD_DELETE = "DELETE"
METHOD_LIST = "LIST"

DEFAULT_LLM_JUDGE_NAME = "llama3.1-70b"
PERSISTED_QUERY_RESULTS_TIMEOUT_IN_MS = (
    20 * 60 * 60 * 1000  # 20 hours (24 hours per Snowflake)
)


class RunDao:
    """Data Access Object for managing AIML RunMetadata entities in Snowflake."""

    def __init__(self, snowpark_session: Session) -> None:
        """Initialize with an active Snowpark session."""
        self.session: Session = snowpark_session

    @staticmethod
    def _compute_invocation_metadata_id(
        dataset_name: str, input_records_count: int
    ):
        return json_utils.obj_id_of_obj(
            obj={
                "dataset_name": dataset_name,
                "input_records_count": input_records_count,
            },
            prefix="invocation_metadata",
        )

    def create_new_run(
        self,
        object_name: str,
        object_type: str,
        run_name: str,
        dataset_name: str,
        source_type: str,
        dataset_spec: dict,
        object_version: Optional[str] = None,
        description: Optional[str] = None,
        label: Optional[str] = None,
        llm_judge_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Create a new RunMetadata entity in Snowflake.

        Args:
            object_name: The name of the managing object for which the run is created under,
                         e.g. name of 'EXTERNAL AGENT'.
            object_type: The type of the managing object. e.g. 'EXTERNAL AGENT'.
            run_name: The name of the run.
            dataset_name: The name of the dataset or user provided dataframe.
            source_type: The type of the source (e.g. 'TABLE').
            dataset_spec: The column specification of the dataset.
            object_version: The version of the managing object.
            description: A description of the run.
            label: A label for the run.
            llm_judge_name: The name of the LLM judge to use for the evaluation, when applicable.

        Returns:
            The result of the Snowflake SQL execution - returning a success message but not the created entity.
        """
        # Build the request payload dictionary.
        req_payload = {
            "object_name": object_name,
            "object_type": object_type,
            "run_name": run_name,
            "description": description,
            "run_metadata": {},
            "source_info": {},
        }

        if object_version:
            req_payload["object_version"] = object_version

        run_metadata_dict = {}

        run_metadata_dict["labels"] = [
            label
        ]  # only accepting a single label for now

        run_metadata_dict["llm_judge_name"] = (
            llm_judge_name if llm_judge_name else DEFAULT_LLM_JUDGE_NAME
        )
        req_payload["run_metadata"] = run_metadata_dict

        source_info_dict = {}
        source_info_dict["name"] = dataset_name
        source_info_dict["column_spec"] = dataset_spec

        if not SourceType.is_valid_source_type(source_type):
            raise ValueError(
                f"Invalid source type: {source_type}. Choose from {SourceType.__members__.values()}"
            )
        source_info_dict["source_type"] = source_type

        req_payload["source_info"] = source_info_dict

        req_payload_json = json.dumps(req_payload)

        query = AIML_RUN_OPS_SYS_FUNC_TEMPLATE.format(method=METHOD_CREATE)

        logger.debug(
            f"Executing query: {query} with parameters {req_payload_json}"
        )

        execute_query(
            self.session,
            query,
            parameters=(req_payload_json,),
        )
        logger.info(
            f"Created new RunMetadata successfully for run '{run_name}'."
        )
        # Re-fetch the newly created run's metadata
        return self.get_run(
            run_name=run_name,
            object_name=object_name,
            object_type=object_type,
            object_version=object_version,
        )

    def get_run(
        self,
        run_name: str,
        object_name: str,
        object_type: str,
        object_version: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retrieve a run.

        Args:
            run_name: The unique name of the run.
            object_name: The managing object's name (e.g. name of EXTERNAL AGENT).
            object_type: The type of the managing object.
            object_version: The version of the managing object.

        Returns:
            A pandas DataFrame containing the run metadata.
        """
        req_payload = {
            "object_name": object_name,
            "object_type": object_type,
            "run_name": run_name,
        }

        if object_version:
            req_payload["object_version"] = object_version

        req_payload_json = json.dumps(req_payload)
        query = AIML_RUN_OPS_SYS_FUNC_TEMPLATE.format(method=METHOD_GET)

        logger.debug(
            f"Executing query: {query} with parameters {req_payload_json}"
        )
        rows: List[Row] = execute_query(
            self.session,
            query,
            parameters=(req_payload_json,),
        )

        if not rows:
            return pd.DataFrame()
        else:
            # Assuming the first row contains our JSON result.
            return pd.DataFrame([rows[0].as_dict()])

    def list_all_runs(self, object_name: str, object_type: str) -> pd.DataFrame:
        """
        List all runs for a given object_name.

        Args:
            object_name: The name of the managing object (e.g. "EXTERNAL AGENT").
            object_type: The type of the managing object.
        Returns:
            A pandas DataFrame containing all run metadata.
        """
        req_payload = {
            "object_name": object_name,
            "object_type": object_type,
        }
        req_payload_json = json.dumps(req_payload)
        query = AIML_RUN_OPS_SYS_FUNC_TEMPLATE.format(method=METHOD_LIST)

        logger.debug(
            f"Executing query: {query} with parameters {req_payload_json}"
        )

        rows: List[Row] = execute_query(
            self.session,
            query,
            parameters=(req_payload_json,),
        )

        return pd.DataFrame([rows[0].as_dict()])

    def _update_run(
        self,
        run_name: str,
        object_name: str,
        object_type: str,
        object_version: Optional[str] = None,
        run_metadata_non_map_field_masks: Optional[List[str]] = None,
        invocation_field_masks: Optional[Dict[str, Any]] = None,
        computation_field_masks: Optional[Dict[str, Any]] = None,
        metric_field_masks: Optional[Dict[str, Any]] = None,
        updated_run_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Generic method to update a run's metadata on the server.

        The payload is built to match the server integration tests. It includes:
          - object_name, object_type, object_version, run_name
          - invocation_field_masks: maps invocation IDs to lists of updated keys (e.g., {"invocation_1": ["id", "input_records_count", "start_time_ms"]})
          - metric_field_masks: similar mapping for metrics (if any)
          - run_metadata_non_map_field_masks: non-map fields to update (e.g. ["labels", "llm_judge_name"])
          - run_metadata: the updated run_metadata (e.g. updated invocations, computations, metrics, etc.)

        Args:
            run_name: The name of the run.
            object_name: The name of the object (e.g. agent name).
            object_type: The type of the object.
            object_version: Optional version string.
            invocation_field_masks: The mask for invocation fields to update.
            metric_field_masks: The mask for metric fields to update.
            updated_run_metadata: The updated run_metadata dict.
        """

        # Build the payload dictionary.
        req_payload = {
            "object_name": object_name,
            "object_type": object_type,
            "run_name": run_name,
            "invocation_field_masks": invocation_field_masks,
            "computation_field_masks": computation_field_masks,
            "metric_field_masks": metric_field_masks,
            "run_metadata_non_map_field_masks": run_metadata_non_map_field_masks,
            "run_metadata": updated_run_metadata,
            "description": "placeholder_description",  # TODO: fix
            "update_description": "false",
        }
        if object_version is not None:
            req_payload["object_version"] = object_version

        req_payload_json = json.dumps(req_payload)

        query = AIML_RUN_OPS_SYS_FUNC_TEMPLATE.format(method="UPDATE")

        logger.debug(
            f"Executing query: {query} with parameters {req_payload_json}"
        )
        execute_query(
            self.session,
            query,
            parameters=(req_payload_json,),
        )

    def upsert_run_metadata_fields(
        self,
        entry_type: str,
        entry_id: str,
        run_name: str,
        object_name: str,
        object_type: str,
        object_version: Optional[str] = None,
        **fields,
    ) -> None:
        """
        Consolidated upsert method for run metadata entries.
        Supported entry types: "invocations", "computations", "metrics".

        This method retrieves the current run metadata, merges new fields with any existing
        entry (if present), and then pushes the updated run metadata via _update_run.

        For "invocations" and "computations", start_time_ms and end_time_ms are required by the backend.
        If these fields are not provided in the new update and the existing entry also does not have them,
        they default to 0. This allows start_time/end_time to be updated from separate calls without
        overwriting previously set values.

        Args:
            entry_type: One of "invocations", "computations", "metrics".
            entry_id: Unique identifier for the metadata entry.
            run_name: The name of the run.
            object_name: The name of the object (e.g. agent name).
            object_type: The type of the object.
            object_version: Optional version.
            **fields: The fields to upsert.
        """
        if entry_type not in SUPPORTED_ENTRY_TYPES:
            raise ValueError(f"Unsupported entry type: {entry_type}")

        # Retrieve the existing run.
        existing_run = self.get_run(
            run_name=run_name,
            object_name=object_name,
            object_type=object_type,
            object_version=object_version,
        )
        if existing_run.empty:
            raise ValueError(f"Run '{run_name}' does not exist.")

        updated_run_metadata = existing_run.get("run_metadata", {})

        # Get the container (e.g., the dictionary for the given entry_type).
        container = updated_run_metadata.get(entry_type, {})

        existing_entry = container.get(entry_id, {})

        # Merge the existing entry with the new fields.
        new_entry = {**existing_entry, **fields}
        new_entry["id"] = entry_id  # Ensure the id is always set.

        container[entry_id] = new_entry
        updated_run_metadata[entry_type] = container

        # TODO: (P0 to fix) ensure we only modify the updated ones
        field_mask = {entry_id: list(new_entry.keys())}

        # field masks for each entry type.
        invocation_field_masks = {}
        metric_field_masks = {}
        computation_field_masks = {}
        if entry_type == SupportedEntryType.INVOCATIONS:
            invocation_field_masks = field_mask
        elif entry_type == SupportedEntryType.COMPUTATIONS:
            computation_field_masks = field_mask
        elif entry_type == SupportedEntryType.METRICS:
            metric_field_masks = field_mask

        self._update_run(
            run_name=run_name,
            object_name=object_name,
            object_type=object_type,
            object_version=object_version,
            invocation_field_masks=invocation_field_masks,
            metric_field_masks=metric_field_masks,
            computation_field_masks=computation_field_masks,
            updated_run_metadata=updated_run_metadata,
        )

    def delete_run(
        self,
        run_name: str,
        object_name: str,
        object_type: str,
        object_version: Optional[str] = None,
    ) -> None:
        """
        Delete a run by its run_name (assumed unique) and object_name.

        Args:
            run_name: The unique name of the run.
            object_name: The managing object's name (e.g. "EXTERNAL AGENT").
            object_type: The type of the managing object.
            object_version: The version of the managing object.
        """
        req_payload = {
            "run_name": run_name,
            "object_name": object_name,
            "object_type": object_type,
        }
        if object_version:
            req_payload["object_version"] = object_version

        req_payload_json = json.dumps(req_payload)

        query = AIML_RUN_OPS_SYS_FUNC_TEMPLATE.format(method=METHOD_DELETE)

        logger.debug(
            f"Executing query: {query} with parameters {req_payload_json}"
        )

        execute_query(
            self.session,
            query,
            parameters=(req_payload_json,),
        )
        logger.info("Deleted run '%s'.", run_name)

    def read_spans_count_from_event_table(
        self, object_name: str, run_name: str, span_type: str
    ) -> int:
        query = """
            SELECT
                COUNT(*) AS record_count
            FROM
                table(snowflake.local.GET_AI_OBSERVABILITY_EVENTS(
                    ?,
                    ?,
                    ?,
                    'EXTERNAL AGENT'
                ))
            WHERE
                RECORD_ATTRIBUTES:"snow.ai.observability.run.name" = ? AND
                RECORD_ATTRIBUTES:"ai.observability.span_type" = ?
        """
        try:
            result_df = self.session.sql(
                query,
                params=[
                    self.session.get_current_database()[1:-1],
                    self.session.get_current_schema()[1:-1],
                    object_name,
                    run_name,
                    span_type,
                ],
            ).to_pandas()

            if "record_count" in result_df.columns:
                count_value = result_df["record_count"].iloc[0]
            else:
                count_value = result_df.iloc[0, 0]
            return int(count_value)
        except Exception as e:
            logger.exception(
                f"Error encountered during reading record count from event table: {e}."
            )
            raise

    def fetch_query_execution_status_by_id(
        self, query_start_time_ms: int, query_id: str
    ) -> str:
        try:
            # NOTE: information_schema.query_history does not always return the latest query status, even within 7 days
            # and account_usage.query_history has higher latency, hence going with snowflake python connector API get_query_status here
            if (
                int(round(time.time() * 1000)) - query_start_time_ms
                > PERSISTED_QUERY_RESULTS_TIMEOUT_IN_MS
            ):
                # https://docs.snowflake.com/en/user-guide/querying-persisted-results

                logger.warning(
                    f"Query {query_id} started almost a day ago, results may not be cached so try to fetch using ACCOUNT_USAGE."
                )
                query = "select EXECUTION_STATUS from snowflake.account_usage.query_history where query_id = ?;"
                ret = self.session.sql(query, params=[query_id]).collect()
                raw_status = ret[0]["EXECUTION_STATUS"]

            else:
                logger.info(
                    "query status using snowflake connector get_query_status()"
                )
                query_status = self.session.connection.get_query_status(
                    query_id
                )

                raw_status = query_status.name

            # resuming_warehouse, running, queued, blocked, success, failed_with_error, or failed_with_incident.
            if "success" in raw_status.lower():
                return "SUCCESS"
            elif "failed" in raw_status.lower():
                return "FAILED"
            else:
                return "IN_PROGRESS"
        except Exception as e:
            logger.exception(
                f"Error encountered during reading query status from query history: {e}."
            )
            raise

    def fetch_computation_job_results_by_query_id(
        self, query_id: str
    ) -> pd.DataFrame:
        curr = self.session.connection.cursor()
        curr.get_results_from_sfqid(query_id)
        return curr.fetch_pandas_all()

    def call_compute_metrics_query(
        self,
        metrics: List[str],
        object_name: str,
        object_version: str,
        object_type: str,
        run_name: str,
    ) -> AsyncJob:
        if not metrics:
            raise ValueError("Metrics list cannot be empty")

        current_db = self.session.get_current_database()
        current_schema = self.session.get_current_schema()

        metrics_str = ",".join([f"'{metric}'" for metric in metrics])
        compute_metrics_query = f"CALL COMPUTE_AI_OBSERVABILITY_METRICS('{current_db}', '{current_schema}', '{object_name}', '{object_version}', '{object_type}', '{run_name}', ARRAY_CONSTRUCT({metrics_str}));"

        compute_query = self.session.sql(compute_metrics_query)

        return compute_query.collect_nowait()
