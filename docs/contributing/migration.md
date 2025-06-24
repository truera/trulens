# ✨ Database Migration

These notes only apply to _TruLens_ developments that change the database
schema.

## Creating a new schema revision

If upgrading the database, you must complete this step.

1. Make desired changes to SQLAlchemy ORM models in `src/core/trulens/core/database/orm.py`.
2. Run automatic Alembic revision script generator. This will generate a new Python script in `src/core/trulens/core/database/migrations`.
   1. `cd src/core/trulens/core/database/migrations`
   2. Set environment variable for database location
      ```bash
      export SQLALCHEMY_URL="sqlite:///../../../../../../default.sqlite"
      ```
   3. Generate migration script
      ```bash
      alembic revision --autogenerate \
        -m "<short_description>" \
        --rev-id "<next_integer_version>"
      ```
3. Check over the automatically generated script in `src/core/trulens/core/database/migrations/versions` to make sure it looks correct.
4. Get a database with the new changes:
   1. `rm default.sqlite`
   2. Run `TruSession()` to create a fresh database that uses the new ORM.
5. Add the version to `src/core/trulens/core/database/migrations/data.py` in the variable `sql_alchemy_migration_versions`
6. Make any `sqlalchemy_upgrade_paths` updates in `src/core/trulens/core/database/migrations/data.py` if a backfill is necessary.

## Creating a DB at the latest schema

If upgrading DB, You must do this step!!

Note: You must create a new schema revision before doing this

Note: Some of these instructions may be outdated and are in the process of being updated.

1. Create a sacrificial OpenAI API key (this will be added to the DB and put it into
   GitHub; which will invalidate it upon commit)
   - **Security Note**: Use a test key that can be safely invalidated, not your production key.
2. cd `tests/docs_notebooks/notebooks_to_test`
3. remove any local DBs
   - `rm -rf default.sqlite`
4. run the below notebooks (Make sure you also run with the most recent code in TruLens) TODO: Move these to a script
   - all_tools.ipynb # `cp ../../../generated_files/all_tools.ipynb ./`
   - llama_index_quickstart.ipynb # `cp ../../../examples/quickstart/llama_index_quickstart.ipynb ./`
   - langchain-retrieval-augmentation-with-trulens.ipynb # `cp ../../../examples/vector-dbs/pinecone/langchain-retrieval-augmentation-with-trulens.ipynb ./`
   - Add any other notebooks you think may introduce possible breaking changes
5. replace the last compatible DB with this new DB file
   - Use the version you chose for --rev-id
   - `mkdir release_dbs/sql_alchemy_<NEW_VERSION>/`
   - `cp default.sqlite release_dbs/sql_alchemy_<NEW_VERSION>/`
6. `git add release_dbs`

## Testing the DB

Run the tests with the requisite environment vars.

```bash
HUGGINGFACE_API_KEY="<to_fill_out>" \
OPENAI_API_KEY="<to_fill_out>" \
PINECONE_API_KEY="<to_fill_out>" \
PINECONE_ENV="<to_fill_out>" \
HUGGINGFACEHUB_API_TOKEN="<to_fill_out>" \
python -m pytest tests/docs_notebooks -k backwards_compat
```
