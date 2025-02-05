import json
import logging
from typing import Any, List

from snowflake.snowpark import Session
import trulens.connectors.snowflake.dao.sql_utils as sql_utils
from trulens.core.run import RunConfig

logger = logging.getLogger(__name__)


AIML_RUN_OPS_SYS_FUNC_TEMPLATE = """
SELECT SYSTEM$AIML_RUN_OPERATION('{method}', ?)
"""
METHOD_CREATE = "CREATE"
METHOD_GET = "GET"
METHOD_UPDATE = "UPDATE"  # TODO
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
    ) -> Any:
        """
        Create a new RunMetadata entity in Snowflake.

        Args:
            object_name: The name of the managing object for which the run is created under,
                         e.g. FQN of 'EXTERNAL_AGENT'.
            object_type: The type of the managing object. e.g. 'EXTERNAL_AGENT'.
            run_name: The name of the run.
            run_config: The configuration for the run.

        Returns:
            The result of the Snowflake SQL execution.
        """
        # Build the request payload dictionary.
        req_payload = {
            "object_name": object_name,
            "object_type": object_type,
            "run_name": run_name,
            "description": run_config.description,
            "label": run_config.label,
            "llm_judge_name": run_config.llm_judge_name,
        }

        # Convert the payload to JSON. You might also use sql_utils.to_json(req_payload)
        # if you have a custom conversion helper.
        req_payload_json = json.dumps(req_payload)

        # Format the query: the method is substituted, while the payload is passed as parameter.
        query = AIML_RUN_OPS_SYS_FUNC_TEMPLATE.format(method=METHOD_CREATE)
        success_message = (
            f"Created new RunMetadata successfully for run '{run_name}'."
        )
        logger.info("Executing query: %s", query)

        # Use sql_utils.execute_query with the JSON payload passed as a parameter tuple.
        sql_utils.execute_query(
            self.session,
            query,
            parameters=(req_payload_json,),
            success_message=success_message,
        )

    def get_run(self, object_name: str, run_name: str) -> Any:
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
        success_message = f"Retrieved run '{run_name}'."
        logger.info("Executing query: %s", query)
        df = sql_utils.fetch_query(
            self.session,
            query,
            success_message=success_message,
            parameters=(req_payload_json,),
        )
        if df.empty:
            return None
        else:
            # Assuming the first row contains our JSON result.
            return df.iloc[0].to_dict()

    def list_all_runs(self, object_name: str, object_type: str) -> List[dict]:
        """
        List all runs for a given object_name.

        Args:
            object_name: The name of the managing object (e.g. "EXTERNAL_AGENT").

        Returns:
            A pandas DataFrame containing the run metadata.
        """
        req_payload = {"object_name": object_name, "object_type": object_type}
        req_payload_json = json.dumps(req_payload)
        query = AIML_RUN_OPS_SYS_FUNC_TEMPLATE.format(method=METHOD_LIST)
        success_message = f"Retrieved list of runs for object '{object_name}'."
        logger.info("Executing query: %s", query)

        df = sql_utils.fetch_query(
            self.session,
            query,
            success_message=success_message,
            parameters=(req_payload_json,),
        )
        if df.empty:
            return []
        else:
            # df contains a list of JSON objects, so we convert each row to a dictionary and make it a list.
            return df.apply(lambda row: row.to_dict(), axis=1).tolist()

    def create_run_if_not_exist(
        self,
        object_name: str,
        object_type: str,
        run_name: str,
        run_config: RunConfig,
    ) -> None:
        """
        Create a new run if one with the given run_name does not already exist.

        Args:
            object_name: The name of the managing object (e.g. "EXTERNAL_AGENT").
            run_name: The name of the run.
            run_config: The configuration for the run.
        """
        run_result = self.get_run(object_name, run_name)
        if run_result is None:
            logger.info("Run '%s' does not exist; creating new run.", run_name)
            self.create_new_run(
                object_name=object_name,
                object_type=object_type,
                run_name=run_name,
                run_config=run_config,
            )
        else:
            logger.info("Run '%s' already exists; skipping creation.", run_name)

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
        success_message = f"Deleted run '{run_name}'."
        logger.info("Executing query: %s", query)
        sql_utils.execute_query(
            self.session,
            query,
            parameters=(req_payload_json,),
            success_message=success_message,
        )
        logger.info("Deleted run '%s'.", run_name)
