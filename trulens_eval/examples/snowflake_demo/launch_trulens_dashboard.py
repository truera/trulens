import os
from dotenv import load_dotenv
from trulens_eval import Tru

load_dotenv()

if __name__ == "__main__":

    db_url = 'snowflake://{user}:{password}@{account}/{dbname}/{schema}?warehouse={warehouse}&role={role}'.format(
        user=os.environ['SF_USER'],
        account=os.environ['SF_ACCOUNT'],
        password=os.environ['SF_PASSWORD'],
        dbname=os.environ['SF_DB_NAME'],
        schema=os.environ['SF_SCHEMA'],
        warehouse=os.environ['SF_WAREHOUSE'],
        role=os.environ['SF_ROLE']
    )
    tru = Tru(database_url=db_url)
    tru.run_dashboard(port=8484)
