import json
import logging
from typing import List, Optional

import pandas as pd
from snowflake.snowpark import Session
from snowflake.snowpark.row import Row
from trulens.connectors.snowflake.dao.sql_utils import execute_query
from trulens.core.run import RunConfig

logger = logging.getLogger(__name__)


AIML_RUN_OPS_SYS_FUNC_TEMPLATE = """
SELECT SYSTEM$AIML_RUN_OPERATION('{method}', ?)
"""
METHOD_CREATE = "CREATE"
METHOD_GET = "GET"
METHOD_UPDATE = "UPDATE"
METHOD_DELETE = "DELETE"
METHOD_LIST = "LIST"


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
        run_config: RunConfig,
    ) -> None:
        """
        Create a new RunMetadata entity in Snowflake.

        Args:
            object_name: The name of the managing object for which the run is created under,
                         e.g. FQN of 'EXTERNAL_AGENT'.
            object_type: The type of the managing object. e.g. 'EXTERNAL_AGENT'.
            run_name: The name of the run.
            run_config: The configuration for the run.

        Returns:
            The result of the Snowflake SQL execution - returning a success message but not the created entity.
        """
        # Build the request payload dictionary.
        req_payload = {
            "object_name": object_name,
            "object_type": object_type,
            "run_name": run_name,
            "description": run_config.description,
            "label": run_config.label,
        }

        # Convert the payload to JSON. You might also use sql_utils.to_json(req_payload)
        # if you have a custom conversion helper.
        req_payload_json = json.dumps(req_payload)

        # Format the query: the method is substituted, while the payload is passed as parameter.
        query = AIML_RUN_OPS_SYS_FUNC_TEMPLATE.format(method=METHOD_CREATE)

        logger.info("Executing query: %s", query)

        # Use sql_utils.execute_query with the JSON payload passed as a parameter tuple.
        execute_query(
            self.session,
            query,
            parameters=(req_payload_json,),
        )
        logger.info(
            f"Created new RunMetadata successfully for run '{run_name}'."
        )

    def get_run(
        self, object_name: str, run_name: str
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve a run by its run_name (assumed unique) and object_name.

        Args:
            object_name: The managing object's name (e.g. "EXTERNAL_AGENT").
            run_name: The unique name of the run.

        Returns:
            A pandas DataFrame containing the run metadata.
        """
        req_payload = {
            "object_name": object_name,
            "run_name": run_name,
        }
        req_payload_json = json.dumps(req_payload)
        query = AIML_RUN_OPS_SYS_FUNC_TEMPLATE.format(method=METHOD_GET)

        logger.info("Executing query: %s", query)
        rows: List[Row] = execute_query(
            self.session,
            query,
            parameters=(req_payload_json,),
        )
        if len(rows) == 0:
            return None
        else:
            # Assuming the first row contains our JSON result.
            return pd.DataFrame([rows[0].as_dict()])

    def list_all_runs(self, object_name: str, object_type: str) -> pd.DataFrame:
        """
        List all runs for a given object_name.

        Args:
            object_name: The name of the managing object (e.g. "EXTERNAL_AGENT").

        Returns:
            A pandas DataFrame containing all run metadata.
        """
        req_payload = {"object_name": object_name, "object_type": object_type}
        req_payload_json = json.dumps(req_payload)
        query = AIML_RUN_OPS_SYS_FUNC_TEMPLATE.format(method=METHOD_LIST)

        logger.info("Executing query: %s", query)

        rows = execute_query(
            self.session,
            query,
            parameters=(req_payload_json,),
        )

        return pd.DataFrame([row.as_dict() for row in rows])

    def create_run_if_not_exist(
        self,
        object_name: str,
        object_type: str,
        run_name: str,
        run_config: RunConfig,
    ) -> Optional[pd.DataFrame]:
        """
        Create a new run if one with the given run_name does not already exist.

        Args:
            object_name: The name of the managing object (e.g. "EXTERNAL_AGENT").
            run_name: The name of the run.
            run_config: The configuration for the run.
        """
        run_result_df = self.get_run(object_name=object_name, run_name=run_name)
        if run_result_df is None:
            logger.info("Run '%s' does not exist; creating new run.", run_name)
            self.create_new_run(
                object_name=object_name,
                object_type=object_type,
                run_name=run_name,
                run_config=run_config,
            )
            logger.info("Created new run '%s' successfully.", run_name)
        else:
            logger.info("Run '%s' already exists; skipping creation.", run_name)

            # Re-fetch the newly created run's metadata
            run_result_df = self.get_run(
                object_name=object_name, run_name=run_name
            )
        return run_result_df

    def delete_run(self, run_name: str, object_name: str, object_type: str):
        """
        Delete a run by its run_name (assumed unique) and object_name.

        Args:
            run_name: The unique name of the run.
            object_name: The managing object's name (e.g. "EXTERNAL_AGENT").
        """
        req_payload = {
            "run_name": run_name,
            "object_name": object_name,
            "object_type": object_type,
        }
        req_payload_json = json.dumps(req_payload)
        query = AIML_RUN_OPS_SYS_FUNC_TEMPLATE.format(method=METHOD_DELETE)

        logger.info("Executing query: %s", query)
        execute_query(
            self.session,
            query,
            parameters=(req_payload_json,),
        )
        logger.info("Deleted run '%s'.", run_name)
