import os

from dotenv import load_dotenv

from trulens_eval import Tru

load_dotenv()

if __name__ == "__main__":

    db_url = 'snowflake://{user}:{password}@{account}/{dbname}/{schema}?warehouse={warehouse}&role={role}'.format(
        user=os.environ['TRULENS_SNOWFLAKE_USER'],
        account=os.environ['TRULENS_SNOWFLAKE_ACCOUNT'],
        password=os.environ['TRULENS_SNOWFLAKE_USER_PASSWORD'],
        dbname=os.environ['TRULENS_SNOWFLAKE_DATABASE'],
        schema=os.environ['TRULENS_SNOWFLAKE_SCHEMA'],
        warehouse=os.environ['TRULENS_SNOWFLAKE_WAREHOUSE'],
        role=os.environ['TRULENS_SNOWFLAKE_ROLE']
    )
    tru = Tru(database_url=db_url)
    tru.run_dashboard(port=8484)
