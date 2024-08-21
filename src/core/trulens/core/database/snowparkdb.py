from typing import Optional
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col
from trulens.core.database.base import DB
import logging
from trulens.core.schema import AppDefinition
from trulens.core.schema.types import FeedbackDefinitionID
import json
import pandas as pd

logger = logging.getLogger(__name__)

class SnowparkDB(DB):
    session: Session = None
    redact_keys: list = []
    class Config:
        arbitrary_types_allowed = True
    def __init__(self, session: Session, redact_keys):
        self.session = session
        self.redact_keys = redact_keys

    def get_app(self, app_id: str):

        """Retrieve an application by its ID."""
        try:
            # Query the App table
            app_df = self.session.table("trulens_apps").filter(col("app_id") == app_id).collect()

            if not app_df:
                logger.warning(f"App with ID {app_id} not found.")
                return None

            app_row = app_df[0]

            # Parse the result into an application object
            return AppDefinition.model_validate(json.loads(app_row['APP_JSON']))

        except Exception as e:
            logger.error(f"Error retrieving app with ID {app_id}: {e}")
            return None

    def get_apps(self):
        
        
        # for _app in session.query(self.orm.AppDefinition):
        #         yield json.loads(_app.app_json)
        # Implement below with snowpark session
        rows = self.session.table("trulens_apps").collect()
        for row in rows:
            yield AppDefinition.model_validate(json.loads(row['APP_JSON']))

    
    def get_feedback_defs(
        self,
        feedback_definition_id: Optional[FeedbackDefinitionID] = None,
    ) -> pd.DataFrame:

        
        rows = self.session.table("trulens_feedback_defs")
        if feedback_definition_id:
            rows = rows.filter(col("feedback_definition_id") == feedback_definition_id)
        fb_defs = rows.collect()
        return pd.DataFrame(
            data=(
            (fb['FEEDBACK_DEFINITION_ID'], json.loads(fb['FEEDBACK_JSON']))
            for fb in fb_defs
            ),
            columns=["feedback_definition_id", "feedback_json"],
        )
    
    def get_feedback(self):
        self.session.table("trulens_feedback").collect()
    def get_feedback_count_by_status(self):
        pass

    def batch_insert_feedback(self):
        pass
    def batch_insert_ground_truth(self):
        pass
    def batch_insert_record(self):
        pass
    def check_db_revision(self):
        pass
    
    def get_datasets(self):
        pass
    def get_ground_truth(self):
        pass
    def get_ground_truths_by_dataset(self):
        pass
    def get_records_and_feedback(self):
        pass
    def insert_app(self):
        pass
    def insert_dataset(self):
        pass
    def insert_feedback(self):
        pass
    def insert_feedback_definition(self):
        pass
    def insert_ground_truth(self):
        pass
    def insert_record(self):
        pass
    def migrate_database(self):
        pass
    def reset_database(self):
        pass