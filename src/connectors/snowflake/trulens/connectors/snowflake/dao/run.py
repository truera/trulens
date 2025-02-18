import json
import logging
from typing import List, Optional

import pandas as pd
from snowflake.snowpark import Session
from snowflake.snowpark.row import Row
from trulens.connectors.snowflake.dao.sql_utils import execute_query

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
DEFAULT_SOURCE_TYPE = "TABLE"


class RunDao:
    """Data Access Object for managing AIML RunMetadata entities in Snowflake."""

    def __init__(self, snowpark_session: Session) -> None:
        """Initialize with an active Snowpark session."""
        self.session: Session = snowpark_session

    def create_new_run(
        self,
        object_name: str,
        object_type: str,
        run_name: str,
        dataset_name: str,
        dataset_col_spec: dict,
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
            dataset_col_spec: The column specification of the dataset.
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
        source_info_dict["column_spec"] = dataset_col_spec
        source_info_dict["source_type"] = (
            DEFAULT_SOURCE_TYPE  # currently only TABLE is supported
        )

        req_payload["source_info"] = source_info_dict

        req_payload_json = json.dumps(req_payload)

        query = AIML_RUN_OPS_SYS_FUNC_TEMPLATE.format(method=METHOD_CREATE)

        logger.info("Executing query: %s", query)

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

        logger.info("Executing query: %s", query)
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

        logger.info("Executing query: %s", query)

        rows: List[Row] = execute_query(
            self.session,
            query,
            parameters=(req_payload_json,),
        )

        return pd.DataFrame([rows[0].as_dict()])

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

        logger.info("Executing query: %s", query)
        execute_query(
            self.session,
            query,
            parameters=(req_payload_json,),
        )
        logger.info("Deleted run '%s'.", run_name)
