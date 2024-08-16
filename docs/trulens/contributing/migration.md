# ✨ Database Migration

These notes only apply to _TruLens_ developments that change the database
schema.

## Creating a new schema revision

If upgrading DB, You must do this step!!

1. Make desired changes to SQLAlchemy orm models in `src/core/trulens/core/database/orm.py`.
1. Get a database with the new changes:
   1. `rm default.sqlite`
   1. Run `Tru()` to create a fresh database that uses the new ORM.
1. Run automatic alembic revision script generator. This will generate a new python script in `src/core/trulens/core/database/migrations`.
   1. `cd src/core/trulens/core/database/migrations`
   1. `SQLALCHEMY_URL="sqlite:///../../../../../../default.sqlite" alembic revision --autogenerate -m "<short_description>" --rev-id "<next_integer_version>"`
1. Check over the automatically generated script in `src/core/trulens/core/database/migration/versions` to make sure it looks correct.
1. Add the version to `src/core/trulens/core/database/migrations/data.py` in the variable `sql_alchemy_migration_versions`
1. Make any `sqlalchemy_upgrade_paths` updates in `src/core/trulens/core/database/migrations/data.py` if a backfill is necessary.

## Creating a DB at the latest schema

If upgrading DB, You must do this step!!

Note: You must create a new schema revision before doing this
Note: Some of these instructions may be outdated and are in progress if being updated.

1. Create a sacrificial OpenAI Key (this will be added to the DB and put into
   github; which will invalidate it upon commit)
1. cd `tests/docs_notebooks/notebooks_to_test`
1. remove any local dbs
    * `rm -rf default.sqlite`
1. run below notebooks (Making sure you also run with the most recent code in
   trulens) TODO: Move these to a script
    * all_tools.ipynb # `cp ../../../generated_files/all_tools.ipynb ./`
    * llama_index_quickstart.ipynb # `cp
      ../../../examples/quickstart/llama_index_quickstart.ipynb ./`
    * langchain-retrieval-augmentation-with-trulens.ipynb # `cp
      ../../../examples/vector-dbs/pinecone/langchain-retrieval-augmentation-with-trulens.ipynb
      ./`
    * Add any other notebooks you think may have possible breaking changes
1. replace the last compatible db with this new db file
    * Use the version you chose for --rev-id
    * `mkdir release_dbs/sql_alchemy_<NEW_VERSION>/`
    * `cp default.sqlite
      release_dbs/sql_alchemy_<NEW_VERSION>/`
1. `git add release_dbs`

## Testing the DB

Run the tests with the requisite env vars.

   ```bash
   HUGGINGFACE_API_KEY="<to_fill_out>" \
   OPENAI_API_KEY="<to_fill_out>" \
   PINECONE_API_KEY="<to_fill_out>" \
   PINECONE_ENV="<to_fill_out>" \
   HUGGINGFACEHUB_API_TOKEN="<to_fill_out>" \
   python -m pytest tests/docs_notebooks -k backwards_compat
   ```
