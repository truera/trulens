# ✨ Database Migration

These notes only apply to _trulens_eval_ developments that change the database
schema.

Warning:
   Some of these instructions may be outdated and are in progress if being updated.

## Creating a new schema revision

If upgrading DB, You must do this step!!

1. `cd truera/trulens_eval/database/migrations`
1. Make sure you have an existing database at the latest schema
    * `mv
      trulens/trulens_eval/release_dbs/sql_alchemy_<LATEST_VERSION>/default.sqlite`
      ./
1. Edit the SQLAlchemy orm models in `trulens_eval/database/orm.py`.
1. Run `export SQLALCHEMY_URL="<url>" && alembic revision --autogenerate -m
   "<short_description>" --rev-id "<next_integer_version>"`
1. Look at the migration script generated at `trulens_eval/database/migration/versions` and edit if
   necessary
1. Add the version to `database/migration/data.py` in variable:
   `sql_alchemy_migration_versions`
1. Make any `data_migrate` updates in `database/migration/data.py` if python changes
   were made
1. `git add truera/trulens_eval/database/migrations/versions`

## Creating a DB at the latest schema

If upgrading DB, You must do this step!!

Note: You must create a new schema revision before doing this

1. Create a sacrificial OpenAI Key (this will be added to the DB and put into
   github; which will invalidate it upon commit)
1. cd `trulens/trulens_eval/tests/docs_notebooks/notebooks_to_test` 
1. remove any local dbs
    * `rm -rf default.sqlite`
1. run below notebooks (Making sure you also run with the most recent code in
   trulens-eval) TODO: Move these to a script
    * all_tools.ipynb # `cp ../../../generated_files/all_tools.ipynb ./`
    * llama_index_quickstart.ipynb # `cp
      ../../../examples/quickstart/llama_index_quickstart.ipynb ./`
    * langchain-retrieval-augmentation-with-trulens.ipynb # `cp
      ../../../examples/vector-dbs/pinecone/langchain-retrieval-augmentation-with-trulens.ipynb
      ./`
    * Add any other notebooks you think may have possible breaking changes
1. replace the last compatible db with this new db file
    * Use the version you chose for --rev-id
    * `mkdir trulens/trulens_eval/release_dbs/sql_alchemy_<NEW_VERSION>/`
    * `cp default.sqlite
      trulens/trulens_eval/release_dbs/sql_alchemy_<NEW_VERSION>/`
1. `git add trulens/trulens_eval/release_dbs`

## Testing the DB

Run the below:

1. `cd trulens/trulens_eval`

2. Run the tests with the requisite env vars.

   ```bash
   HUGGINGFACE_API_KEY="<to_fill_out>" \
   OPENAI_API_KEY="<to_fill_out>" \
   PINECONE_API_KEY="<to_fill_out>" \
   PINECONE_ENV="<to_fill_out>" \
   HUGGINGFACEHUB_API_TOKEN="<to_fill_out>" \
   python -m pytest tests/docs_notebooks -k backwards_compat
   ```
