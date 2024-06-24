import os

from dotenv import load_dotenv

from trulens_eval import Tru

load_dotenv()

if __name__ == "__main__":

    db_url = 'snowflake://{user}:{password}@{account}/{dbname}/{schema}?warehouse={warehouse}&role={role}'.format(
        user=os.environ['SNOWFLAKE_USER'],
        account=os.environ['SNOWFLAKE_ACCOUNT'],
        password=os.environ['SNOWFLAKE_USER_PASSWORD'],
        dbname=os.environ['SNOWFLAKE_DATABASE'],
        schema=os.environ['SNOWFLAKE_SCHEMA'],
        warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
        role=os.environ['SNOWFLAKE_ROLE']
    )
    tru = Tru(database_url=db_url)
    tru.run_dashboard(port=8484)
