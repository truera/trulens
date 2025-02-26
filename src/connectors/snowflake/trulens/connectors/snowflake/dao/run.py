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

DEFAULT_LLM_JUDGE_NAME = "llama3.1-70b"


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
            "computation_field_masks": computation_field_masks,
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
        # Retrieve the existing run.
        existing_run = self.get_run(
            run_name=run_name,
            object_name=object_name,
            object_type=object_type,
            object_version=object_version,
        )
        if existing_run.empty:
            raise ValueError(f"Run '{run_name}' does not exist.")

        # Get the current run_metadata (or default to empty dict).
        updated_run_metadata = existing_run.get("run_metadata", {})

        # Ensure the entry_type is supported.
        if entry_type not in ["invocations", "computations", "metrics"]:
            raise ValueError(f"Unsupported entry type: {entry_type}")

        # Get the container (e.g., the dictionary for the given entry_type).
        container = updated_run_metadata.get(entry_type, {})

        # Retrieve any existing entry; if none, default to an empty dict.
        existing_entry = container.get(entry_id, {})

        # Merge the existing entry with the new fields.
        new_entry = {**existing_entry, **fields}
        new_entry["id"] = entry_id  # Ensure the id is always set.

        # Update the container with the merged entry.
        container[entry_id] = new_entry
        updated_run_metadata[entry_type] = container

        # Build the field mask from the keys of the new entry.
        field_mask = {entry_id: list(new_entry.keys())}

        # Prepare masks for each type.
        invocation_field_masks = {}
        metric_field_masks = {}
        computation_field_masks = {}
        if entry_type == "invocations":
            invocation_field_masks = field_mask
        elif entry_type == "computations":
            computation_field_masks = field_mask
        elif entry_type == "metrics":
            metric_field_masks = field_mask

        # Push the updated run metadata.
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
