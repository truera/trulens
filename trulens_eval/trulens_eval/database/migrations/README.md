# Database Migrations
Database schema revisions are handled with [Alembic](https://github.com/sqlalchemy/alembic/)

## Upgrading to the latest schema revision

```python
from trulens_eval import Tru

tru = Tru(database_url="<sqlalchemy_url>")
tru.migrate_database()
```


## Creating a new schema revision
If upgrading DB, You must do this step!!

1. `cd truera/trulens_eval/database/migrations`
1. Make sure you have an existing database at the latest schema
    * `mv trulens/trulens_eval/release_dbs/sql_alchemy_<LATEST_VERSION>/default.sqlite` ./
1. Edit the [SQLAlchemy models](../orm.py)
1. Run `export SQLALCHEMY_URL="<url>" && alembic revision --autogenerate -m "<short_description>" --rev-id "<next_integer_version>"`
1. Look at the migration script generated at [versions](./versions) and edit if necessary
1. Add the version to `db_data_migration.py` in variable: `sql_alchemy_migration_versions`
1. Make any `data_migrate` updates in `db_data_migration.py` if python changes were made
1. `git add truera/trulens_eval/database/migrations/versions`

## Creating a DB at the latest schema
If upgrading DB, You must do this step!!

Note: You must create a new schema revision before doing this

1. Create a sacrificial OpenAI Key (this will be added to the DB and put into github; which will invalidate it upon commit)
1. cd `trulens/trulens_eval/tests/docs_notebooks/notebooks_to_test` 
1. remove any local dbs
    * `rm -rf default.sqlite`
1. run below notebooks (Making sure you also run with the most recent code in trulens-eval) TODO: Move these to a script
    * all_tools.ipynb # `cp ../../../generated_files/all_tools.ipynb ./`
    * llama_index_quickstart.ipynb # `cp ../../../examples/frameworks/llama_index/llama_index_quickstart.ipynb ./`
    * langchain-retrieval-augmentation-with-trulens.ipynb # `cp ../../../examples/vector-dbs/pinecone/langchain-retrieval-augmentation-with-trulens.ipynb ./`
    * Add any other notebooks you think may have possible breaking changes
1. replace the last compatible db with this new db file
    * Use the version you chose for --rev-id
    * `mkdir trulens/trulens_eval/release_dbs/sql_alchemy_<NEW_VERSION>/`
    * `cp default.sqlite trulens/trulens_eval/release_dbs/sql_alchemy_<NEW_VERSION>/`
1. `git add trulens/trulens_eval/release_dbs`

## Testing the DB
Run the below:
1. `cd trulens/trulens_eval`
1. `HUGGINGFACE_API_KEY="<to_fill_out>"  OPENAI_API_KEY="<to_fill_out>" PINECONE_API_KEY="" PINECONE_ENV="" HUGGINGFACEHUB_API_TOKEN="" python -m pytest tests/docs_notebooks -k backwards_compat`

## Copying a database
Have a look at the help text for `_copy_database` and take into
account all the items under the section `Important considerations`:

```python

from trulens_eval.database.utils import _copy_database

help(_copy_database)
```

Copy all data from the source database into an EMPTY target database:

```python

from trulens_eval.database.utils import _copy_database

_copy_database(
    src_url="<source_db_url>",
    tgt_url="<target_db_url>"
)
```
