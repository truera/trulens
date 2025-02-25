import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from snowflake.snowpark import Session
from snowflake.snowpark.row import Row
from trulens.connectors.snowflake.dao.enums import SourceType
from trulens.connectors.snowflake.dao.sql_utils import execute_query
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

DEFAULT_LLM_JUDGE_NAME = "mistral-large2"


class RunDao:
    """Data Access Object for managing AIML RunMetadata entities in Snowflake."""

    def __init__(self, snowpark_session: Session) -> None:
        """Initialize with an active Snowpark session."""
        self.session: Session = snowpark_session

    # @staticmethod
    # def _compute_source_info_id(dataset_name: str, dataset_spec: dict) -> str:
    #     return json_utils.obj_id_of_obj(
    #         obj={"dataset_name": dataset_name, "dataset_spec": dataset_spec},
    #         prefix="source_info",
    #     )

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

    @staticmethod
    def _generate_computation_metadata_id():
        # generate a random UUID for the computation metadata
        pass

    @staticmethod
    def _compute_metrics_metadata_id():
        pass

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
        req_payload = {"object_name": object_name, "object_type": object_type}
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
          - run_metadata: the complete, updated run_metadata (e.g. updated invocations, computations, metrics, etc.)
          - description: "new_description"
          - update_description: "true"

        Args:
            run_name: The name of the run.
            object_name: The name of the object (e.g. agent name).
            object_type: The type of the object.
            object_version: Optional version string.
            invocation_field_masks: The mask for invocation fields to update.
            metric_field_masks: The mask for metric fields to update.
            updated_run_metadata: The complete updated run_metadata dict.
        """

        # Build the payload dictionary.
        payload = {
            "object_name": object_name,
            "object_type": object_type,
            "run_name": run_name,
            "invocation_field_masks": invocation_field_masks,
            "metric_field_masks": metric_field_masks,
            "run_metadata_non_map_field_masks": run_metadata_non_map_field_masks,
            "run_metadata": updated_run_metadata,
            "description": "new_description",
            "update_description": "true",
        }
        if object_version is not None:
            payload["object_version"] = object_version

        req_payload_json = json.dumps(payload)
        query = AIML_RUN_OPS_SYS_FUNC_TEMPLATE.format(method="UPDATE")

        logger.debug(
            f"Executing query: {query} with parameters {req_payload_json}"
        )
        execute_query(
            self.session,
            query,
            parameters=(req_payload_json,),
        )

    def upsert_invocation_metadata(
        self,
        invocation_metadata_id: str,
        run_name: str,
        object_name: str,
        object_type: str,
        input_records_count: Optional[int] = None,
        object_version: Optional[str] = None,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        completion_status: Optional[Dict] = None,
    ):
        existing_run: pd.DataFrame = self.get_run(
            run_name=run_name,
            object_name=object_name,
            object_type=object_type,
            object_version=object_version,
        )
        if existing_run.empty:
            raise ValueError(f"Run '{run_name}' does not exist.")

        updated_run_metadata = existing_run.get("run_metadata", {})

        logger.debug(f"Existing run metadata: {updated_run_metadata}")

        invocations = updated_run_metadata.get("invocations", {})

        # Check if the invocation already exists.
        if invocation_metadata_id in invocations:
            existing_invocation = invocations[invocation_metadata_id]
            # Update the existing invocation with new values.
            if input_records_count is not None:
                existing_invocation["input_records_count"] = input_records_count
            if start_time_ms is not None:
                existing_invocation["start_time_ms"] = start_time_ms
            if end_time_ms is not None:
                existing_invocation["end_time_ms"] = end_time_ms
            if completion_status is not None:
                existing_invocation["completion_status"] = completion_status
        else:
            # Create a new invocation entry with only the required fields.
            new_invocation = {
                "id": invocation_metadata_id,
                "input_records_count": input_records_count,
            }
            if start_time_ms:
                new_invocation["start_time_ms"] = start_time_ms
            else:
                new_invocation["start_time_ms"] = (
                    0  # start and end time is always required by DPO
                )
            if end_time_ms:
                new_invocation["end_time_ms"] = end_time_ms
            else:
                new_invocation["end_time_ms"] = (
                    0  # end time is always required by DPO
                )
            if completion_status:
                new_invocation["completion_status"] = completion_status

            # Update the invocations dictionary.
            invocations[invocation_metadata_id] = new_invocation

        # Update the invocations dictionary.
        invocations[invocation_metadata_id] = new_invocation
        updated_run_metadata["invocations"] = invocations

        # The field mask tells the server which keys in the invocation entry are updated.
        invocation_field_masks = {
            invocation_metadata_id: list(new_invocation.keys())
        }
        metric_field_masks = {}  # No metric updates in this operation.

        # Push the updated run metadata to the server.
        self._update_run(
            run_name=run_name,
            object_name=object_name,
            object_type=object_type,
            object_version=object_version,
            invocation_field_masks=invocation_field_masks,
            metric_field_masks=metric_field_masks,
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
