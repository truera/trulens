import abc
from datetime import datetime
import json
import logging
from pathlib import Path
from pprint import PrettyPrinter
import sqlite3
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

from merkle_json import MerkleJson
import numpy as np
import pandas as pd
import pydantic

from trulens_eval import __version__
from trulens_eval.database.legacy import migration
from trulens_eval.app import App
from trulens_eval.database.legacy.migration import MIGRATION_UNKNOWN_STR
from trulens_eval.feedback import Feedback
from trulens_eval.schema import AppDefinition
from trulens_eval.schema import AppID
from trulens_eval.schema import Cost
from trulens_eval.schema import FeedbackDefinition
from trulens_eval.schema import FeedbackDefinitionID
from trulens_eval.schema import FeedbackResult
from trulens_eval.schema import FeedbackResultID
from trulens_eval.schema import FeedbackResultStatus
from trulens_eval.schema import Perf
from trulens_eval.schema import Record
from trulens_eval.schema import RecordID
from trulens_eval.utils.json import json_str_of_obj
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.serial import JSONized
from trulens_eval.utils.serial import SerialModel
from trulens_eval.utils.text import UNICODE_CHECK
from trulens_eval.utils.text import UNICODE_CLOCK

mj = MerkleJson()
NoneType = type(None)

pp = PrettyPrinter()

logger = logging.getLogger(__name__)

MULTI_CALL_NAME_DELIMITER = ":::"

DEFAULT_DATABASE_PREFIX: str = "trulens_"
"""Default prefix for table names for trulens_eval to use.

This includes alembic's version table.
"""

DEFAULT_DATABASE_FILE: str = "default.sqlite"
"""Filename for default sqlite database.

The sqlalchemy url for this default local sqlite database is `sqlite:///default.sqlite`.
"""

DEFAULT_DATABASE_REDACT_KEYS: bool = False
"""Default value for option to redact secrets before writing out data to database."""

class DBMeta(pydantic.BaseModel):
    """
    Database meta data mostly used for migrating from old db schemas.
    """

    trulens_version: Optional[str]
    attributes: dict


class DB(SerialModel, abc.ABC):
    """Abstract definition of databases used by trulens_eval.
    
    [SqlAlchemyDB][trulens_eval.database.sqlalchemy_db.SqlAlchemyDB] is the main
    and default implementation if this specification.
    """

    redact_keys: bool = False
    """Redact secrets before writing out data."""

    table_prefix: str = "trulens_"
    """Prefix for table names for trulens_eval to use.
    
    May be useful in some databases where trulens is not the only app.
    """

    def _json_str_of_obj(self, obj: Any) -> str:
        return json_str_of_obj(obj, redact_keys=self.redact_keys)

    @abc.abstractmethod
    def reset_database(self):
        """Delete all data."""

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_record(
        self,
        record: Record,
    ) -> RecordID:
        """
        Upsert a `record` into the database.
        
        Args:
            record: The record to insert or update.

        Returns:
            The id of the given record.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_app(self, app: AppDefinition) -> AppID:
        """
        Upsert an `app` into the database.

        Args:
            app: The app to insert or update. Note that only the
                [AppDefinition][trulens_eval.schema.AppDefinition] parts are serialized
                hence the type hint.

        Returns:
            The id of the given app.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_feedback_definition(
        self, feedback_definition: FeedbackDefinition
    ) -> FeedbackDefinitionID:
        """
        Upsert a `feedback_definition` into the databaase.

        Args:
            feedback_definition: The feedback definition to insert or update.
                Note that only the
                [FeedbackDefinition][trulens_eval.schema.FeedbackDefinition]
                parts are serialized hence the type hint.

        Returns:
            The id of the given feedback definition.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def get_feedback_defs(
        self, feedback_definition_id: Optional[FeedbackDefinitionID] = None
    ) -> pd.DataFrame:
        """Retrieve feedback definitions from the database.
        
        Args:
            feedback_definition_id: if provided, only the
                feedback definition with the given id is returned. Otherwise,
                all feedback definitions are returned.

        Returns:
            A dataframe with the feedback definitions.
        """


        raise NotImplementedError()

    @abc.abstractmethod
    def insert_feedback(
        self,
        feedback_result: FeedbackResult,
    ) -> FeedbackResultID:
        """Upsert a `feedback_result` into the the database.

        Args:
            feedback_result: The feedback result to insert or update.

        Returns:
            The id of the given feedback result.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def get_feedback(
        self,
        record_id: Optional[RecordID] = None,
        feedback_result_id: Optional[FeedbackResultID] = None,
        feedback_definition_id: Optional[FeedbackDefinitionID] = None,
        status: Optional[Union[FeedbackResultStatus,
                               Sequence[FeedbackResultStatus]]] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        shuffle: Optional[bool] = None
    ) -> pd.DataFrame:
        """Get feedback results matching a set of optional criteria:

        Args:
            record_id: Get only the feedback for the given record id.

            feedback_result_id: Get only the feedback for the given feedback
                result id.

            feedback_definition_id: Get only the feedback for the given feedback
                definition id.

            status: Get only the feedback with the given status. If a sequence
                of statuses is given, all feedback with any of the given
                statuses are returned.

            last_ts_before: get only results with `last_ts` before the
                given datetime.

            offset: index of the first row to return.

            limit: limit the number of rows returned.

            shuffle: shuffle the rows before returning them.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def get_feedback_count_by_status(
        self,
        record_id: Optional[RecordID] = None,
        feedback_result_id: Optional[FeedbackResultID] = None,
        feedback_definition_id: Optional[FeedbackDefinitionID] = None,
        status: Optional[Union[FeedbackResultStatus,
                               Sequence[FeedbackResultStatus]]] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        shuffle: bool = False
    ) -> Dict[FeedbackResultStatus, int]:
        """Get count of feedback results matching a set of optional criteria grouped by
        their status.
        
        See [get_feedback][trulens_eval.database.base.DB.get_feedback] for the meaning of
        the the arguments.

        Returns:
            A mapping of status to the count of feedback results of that status
                that match the given filters.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def get_app(self, app_id: AppID) -> Optional[JSONized[App]]:
        """Get the app with the given id from the database.
        
        Returns:
            The jsonized version of the app with the given id. Deserialization
                can be done with
                [App.model_validate][trulens_eval.app.App.model_validate].
        
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_records_and_feedback(
        self,
        app_ids: Optional[List[AppID]] = None
    ) -> Tuple[pd.DataFrame, Sequence[str]]:
        """Get records fom the database.
        
        Args:
            app_ids: If given, retrieve only the records for the given apps.
                Otherwise all apps are retrieved.
        
        Returns:
            A dataframe with the records.

            A list of column names that contain feedback results.
        """
        raise NotImplementedError()


def versioning_decorator(func):
    """A function decorator that checks if a DB is up to date.
     
    Check that the database is up to date with the current
    trulens_eval version (or rather the db version used in the current
    trulens_eval version).
    """
    
    def wrapper(self, *args, **kwargs):
        migration._migration_checker(db=self)
        returned_value = func(self, *args, **kwargs)
        return returned_value

    return wrapper


def for_all_methods(decorator):
    """
    A class decorator that will decorate all DB Access methods except for
    instantiations, db resets, or version checking.
    """

    def decorate(cls):
        for attr in cls.__dict__:
            if not str(attr).startswith("_") and str(attr) not in [
                    "get_meta", "reset_database", "migrate_database"
            ] and callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate


@for_all_methods(versioning_decorator)
class LocalSQLite(DB):
    """
    Warning:
        This database implementation is deprecated and will be removed in the
        future. Use
        [SqlAlchemyDB][trulens_eval.database.sqlalchemy_db.SqlAlchemyDB]
        instead. Local sqlite databases can be initialized with SqlAlchemyDB.
    """

    filename: Path

    TABLE_META: ClassVar[str] = "meta"
    TABLE_RECORDS: ClassVar[str] = "records"
    TABLE_FEEDBACKS: ClassVar[str] = "feedbacks"
    TABLE_FEEDBACK_DEFS: ClassVar[str] = "feedback_defs"
    TABLE_APPS: ClassVar[str] = "apps"

    TYPE_TIMESTAMP: ClassVar[str] = "FLOAT"
    TYPE_ENUM: ClassVar[str] = "TEXT"
    TYPE_JSON: ClassVar[str] = "TEXT"

    TABLES: ClassVar[List[str]] = [
        TABLE_RECORDS, TABLE_FEEDBACKS, TABLE_FEEDBACK_DEFS, TABLE_APPS
    ]

    def __init__(self, filename: Path, redact_keys: bool = False):
        """
        Database locally hosted using SQLite.

        Args

        - filename: Optional[Path] -- location of sqlite database dump file. It
          will be created if it does not exist.
        - redact_keys: bool -- redact secret keys before writing anything to the
          db. What is and is not a secret key is determined by
          `keys.py:should_redact_{key, value}`.

        """
        super().__init__(filename=filename, redact_keys=redact_keys)

        self._build_tables()
        migration._migration_checker(db=self, warn=True)

    def __str__(self) -> str:
        return f"SQLite({self.filename})"

    # DB requirement
    def reset_database(self) -> None:
        self._drop_tables()
        self._build_tables()

    def migrate_database(self, prior_table_prefix: str = ""):
        """
        Migrate the database to the current version of trulens_eval. If the
        prior version used a different name for the version table, it must be
        provided to this method.

        Args:
            prior_table_prefix: The table name prefix used in the prior version
                of the database or the database we need to migrate from.
        """

        # TODO: figure out how to update the version table alone if that is all
        # that needs to happen

        migration.migrate(db=self)

    def _clear_tables(self) -> None:
        conn, c = self._connect()

        for table in self.TABLES:
            c.execute(f'''DELETE FROM {table}''')

        self._close(conn)

    def _drop_tables(self) -> None:
        conn, c = self._connect()

        for table in self.TABLES:
            c.execute(f'''DROP TABLE IF EXISTS {table}''')

        self._close(conn)

    def get_meta(self):
        conn, c = self._connect()

        try:
            c.execute(f'''SELECT key, value FROM {self.TABLE_META}''')
            rows = c.fetchall()
            ret = {}

            for row in rows:
                ret[row[0]] = row[1]

            if 'trulens_version' in ret:
                trulens_version = ret['trulens_version']
            else:
                trulens_version = None

            return DBMeta(trulens_version=trulens_version, attributes=ret)

        except Exception as e:
            return DBMeta(trulens_version=None, attributes={})

    def _create_db_meta_table(self, c):
        c.execute(
            f'''CREATE TABLE IF NOT EXISTS {self.TABLE_META} (
                key TEXT NOT NULL PRIMARY KEY,
                value TEXT
            )'''
        )
        # Create table if it does not exist. Note that the record_json column
        # also encodes inside it all other columns.

        meta = self.get_meta()

        if meta.trulens_version is None:
            db_version = __version__
            c.execute(
                f"""
                SELECT name FROM sqlite_master  
                WHERE type='table';
                """
            )
            rows = c.fetchall()

            if len(rows) > 1:
                # _create_db_meta_table is called before any DB manipulations,
                # so if existing tables are present but it's an empty metatable, it means this is trulens-eval first release.
                db_version = "0.1.2"
            # Otherwise, set the version
            c.execute(
                f'''INSERT INTO {self.TABLE_META} VALUES (?, ?)''',
                ('trulens_version', db_version)
            )

    def _build_tables(self):
        conn, c = self._connect()
        self._create_db_meta_table(c)
        c.execute(
            f'''CREATE TABLE IF NOT EXISTS {self.TABLE_RECORDS} (
                record_id TEXT NOT NULL PRIMARY KEY,
                app_id TEXT NOT NULL,
                input TEXT,
                output TEXT,
                record_json {self.TYPE_JSON} NOT NULL,
                tags TEXT NOT NULL,
                ts {self.TYPE_TIMESTAMP} NOT NULL,
                cost_json {self.TYPE_JSON} NOT NULL,
                perf_json {self.TYPE_JSON} NOT NULL
            )'''
        )
        c.execute(
            f'''CREATE TABLE IF NOT EXISTS {self.TABLE_FEEDBACKS} (
                feedback_result_id TEXT NOT NULL PRIMARY KEY,
                record_id TEXT NOT NULL,
                feedback_definition_id TEXT,
                last_ts {self.TYPE_TIMESTAMP} NOT NULL,
                status {self.TYPE_ENUM} NOT NULL,
                error TEXT,
                calls_json {self.TYPE_JSON} NOT NULL,
                result FLOAT,
                name TEXT NOT NULL,
                cost_json {self.TYPE_JSON} NOT NULL,
                multi_result {self.TYPE_JSON}
            )'''
        )
        c.execute(
            f'''CREATE TABLE IF NOT EXISTS {self.TABLE_FEEDBACK_DEFS} (
                feedback_definition_id TEXT NOT NULL PRIMARY KEY,
                feedback_json {self.TYPE_JSON} NOT NULL
            )'''
        )
        c.execute(
            f'''CREATE TABLE IF NOT EXISTS {self.TABLE_APPS} (
                app_id TEXT NOT NULL PRIMARY KEY,
                app_json {self.TYPE_JSON} NOT NULL
            )'''
        )
        self._close(conn)

    def _connect(self) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        return conn, c

    def _close(self, conn: sqlite3.Connection) -> None:
        conn.commit()
        conn.close()

    # DB requirement
    def insert_record(
        self,
        record: Record,
    ) -> RecordID:
        # NOTE: Oddness here in that the entire record is put into the
        # record_json column while some parts of that records are also put in
        # other columns. Might want to keep this so we can query on the columns
        # within sqlite.

        vals = (
            record.record_id, record.app_id,
            self._json_str_of_obj(record.main_input),
            self._json_str_of_obj(record.main_output),
            self._json_str_of_obj(record), record.tags, record.ts,
            self._json_str_of_obj(record.cost),
            self._json_str_of_obj(record.perf)
        )

        self._insert_or_replace_vals(table=self.TABLE_RECORDS, vals=vals)

        print(
            f"{UNICODE_CHECK} record {record.record_id} from {record.app_id} -> {self.filename}"
        )

        return record.record_id

    # DB requirement
    def insert_app(self, app: AppDefinition) -> AppID:
        app_id = app.app_id
        app_str = app.model_dump_json()

        vals = (app_id, app_str)
        self._insert_or_replace_vals(table=self.TABLE_APPS, vals=vals)

        print(f"{UNICODE_CHECK} app {app_id} -> {self.filename}")

        return app_id

    def insert_feedback_definition(
        self, feedback: Union[Feedback, FeedbackDefinition]
    ) -> FeedbackDefinitionID:
        """
        Insert a feedback definition into the database.
        """

        feedback_definition_id = feedback.feedback_definition_id
        feedback_str = feedback.model_dump_json()
        vals = (feedback_definition_id, feedback_str)

        self._insert_or_replace_vals(table=self.TABLE_FEEDBACK_DEFS, vals=vals)

        print(
            f"{UNICODE_CHECK} feedback def. {feedback_definition_id} -> {self.filename}"
        )

        return feedback_definition_id

    def get_feedback_defs(
        self, feedback_definition_id: Optional[str] = None
    ) -> pd.DataFrame:

        clause = ""
        args = ()
        if feedback_definition_id is not None:
            clause = "WHERE feedback_id=?"
            args = (feedback_definition_id,)

        query = f"""
            SELECT
                feedback_definition_id, feedback_json
            FROM {self.TABLE_FEEDBACK_DEFS}
            {clause}
        """

        conn, c = self._connect()
        c.execute(query, args)
        rows = c.fetchall()
        self._close(conn)

        df = pd.DataFrame(
            rows, columns=[description[0] for description in c.description]
        )

        return df

    def _insert_or_replace_vals(self, table, vals):
        conn, c = self._connect()
        c.execute(
            f"""INSERT OR REPLACE INTO {table}
                VALUES ({','.join('?' for _ in vals)})""", vals
        )
        self._close(conn)

    def insert_feedback(
        self, feedback_result: FeedbackResult
    ) -> FeedbackResultID:
        """
        Insert a record-feedback link to db or update an existing one.
        """

        vals = (
            feedback_result.feedback_result_id,
            feedback_result.record_id,
            feedback_result.feedback_definition_id,
            feedback_result.last_ts.timestamp(),
            feedback_result.status.value,
            feedback_result.error,
            self._json_str_of_obj(
                dict(calls=feedback_result.calls)
            ),  # extra dict is needed json's root must be a dict
            feedback_result.result,
            feedback_result.name,
            self._json_str_of_obj(feedback_result.cost),
            feedback_result.multi_result
        )

        self._insert_or_replace_vals(table=self.TABLE_FEEDBACKS, vals=vals)

        if feedback_result.status == FeedbackResultStatus.DONE:
            print(
                f"{UNICODE_CHECK} feedback {feedback_result.feedback_result_id} on {feedback_result.record_id} -> {self.filename}"
            )
        else:
            print(
                f"{UNICODE_CLOCK} feedback {feedback_result.feedback_result_id} on {feedback_result.record_id} -> {self.filename}"
            )

    def get_feedback(
        self,
        record_id: Optional[RecordID] = None,
        feedback_result_id: Optional[FeedbackResultID] = None,
        feedback_definition_id: Optional[FeedbackDefinitionID] = None,
        status: Optional[FeedbackResultStatus] = None,
        last_ts_before: Optional[datetime] = None
    ) -> pd.DataFrame:

        clauses = []
        vars = []

        if record_id is not None:
            clauses.append("f.record_id=?")
            vars.append(record_id)

        if feedback_result_id is not None:
            clauses.append("f.feedback_result_id=?")
            vars.append(feedback_result_id)

        if feedback_definition_id is not None:
            clauses.append("f.feedback_definition_id=?")
            vars.append(feedback_definition_id)

        if status is not None:
            if isinstance(status, Sequence):
                clauses.append(
                    "f.status in (" + (",".join(["?"] * len(status))) + ")"
                )
                for v in status:
                    vars.append(v.value)
            else:
                clauses.append("f.status=?")
                vars.append(status)

        if last_ts_before is not None:
            clauses.append("f.last_ts<=?")
            vars.append(last_ts_before.timestamp())

        where_clause = " AND ".join(clauses)
        if len(where_clause) > 0:
            where_clause = " AND " + where_clause

        query = f"""
            SELECT
                f.record_id, f.feedback_result_id, f.feedback_definition_id, 
                f.last_ts,
                f.status,
                f.error,
                f.name as fname,
                f.result, 
                f.cost_json,
                r.perf_json,
                f.calls_json,
                fd.feedback_json, 
                r.record_json, 
                c.app_json
            FROM {self.TABLE_RECORDS} r
                JOIN {self.TABLE_FEEDBACKS} f 
                JOIN {self.TABLE_FEEDBACK_DEFS} fd
                JOIN {self.TABLE_APPS} c
            WHERE f.feedback_definition_id=fd.feedback_definition_id
                AND r.record_id=f.record_id
                AND r.app_id=c.app_id
                {where_clause}
        """

        conn, c = self._connect()
        c.execute(query, vars)
        rows = c.fetchall()
        self._close(conn)

        df = pd.DataFrame(
            rows, columns=[description[0] for description in c.description]
        )

        def map_row(row):
            # NOTE: pandas dataframe will take in the various classes below but the
            # agg table used in UI will not like it. Sending it JSON/dicts instead.

            row.calls_json = json.loads(
                row.calls_json
            )['calls']  # calls_json (sequence of FeedbackCall)
            row.cost_json = json.loads(row.cost_json)  # cost_json (Cost)
            try:
                # Add a try-catch here as latency is a DB breaking change, but not a functionality breaking change.
                # If it fails, we can still continue.
                row.perf_json = json.loads(row.perf_json)  # perf_json (Perf)
                row['latency'] = Perf(**row.perf_json).latency
            except:
                # If it comes here, it is because we have filled the DB with a migration tag that cannot be loaded into perf_json
                # This is not migrateable because start/end times were not logged and latency is required, but adding a real latency
                # would create incorrect summations
                pass
            row.feedback_json = json.loads(
                row.feedback_json
            )  # feedback_json (FeedbackDefinition)
            row.record_json = json.loads(
                row.record_json
            )  # record_json (Record)
            row.app_json = json.loads(row.app_json)  # app_json (App)
            app = AppDefinition(**row.app_json)

            row.status = FeedbackResultStatus(row.status)

            row['total_tokens'] = row.cost_json['n_tokens']
            row['total_cost'] = row.cost_json['cost']

            row['type'] = app.root_class

            return row

        df = df.apply(map_row, axis=1)
        return pd.DataFrame(df)

    def get_app(self, app_id: str) -> JSON:
        conn, c = self._connect()
        c.execute(
            f"SELECT app_json FROM {self.TABLE_APPS} WHERE app_id=?", (app_id,)
        )
        result = c.fetchone()[0]
        conn.close()

        return json.loads(result)

    def get_records_and_feedback(
        self,
        app_ids: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Sequence[str]]:
        # This returns all apps if the list of app_ids is empty.
        app_ids = app_ids or []

        conn, c = self._connect()
        query = f"""
            SELECT r.record_id, f.calls_json, f.result, f.name, f.multi_result
            FROM {self.TABLE_RECORDS} r 
            LEFT JOIN {self.TABLE_FEEDBACKS} f
                ON r.record_id = f.record_id
            """
        if len(app_ids) > 0:
            app_id_list = ', '.join('?' * len(app_ids))
            query = query + f" WHERE r.app_id IN ({app_id_list})"

        c.execute(query, app_ids)
        rows = c.fetchall()
        conn.close()
        df_results = pd.DataFrame(
            rows, columns=[description[0] for description in c.description]
        )
        if len(df_results) == 0:
            return df_results, []

        conn, c = self._connect()
        query = f"""
            SELECT DISTINCT r.*, c.app_json
            FROM {self.TABLE_RECORDS} r 
            JOIN {self.TABLE_APPS} c
                ON r.app_id = c.app_id
            """
        if len(app_ids) > 0:
            app_id_list = ', '.join('?' * len(app_ids))
            query = query + f" WHERE r.app_id IN ({app_id_list})"

        c.execute(query, app_ids)
        rows = c.fetchall()
        conn.close()

        df_records = pd.DataFrame(
            rows, columns=[description[0] for description in c.description]
        )

        apps = df_records['app_json'].apply(AppDefinition.model_validate_json)
        df_records['type'] = apps.apply(lambda row: str(row.root_class))

        cost = df_records['cost_json'].map(Cost.model_validate_json)
        df_records['total_tokens'] = cost.map(lambda v: v.n_tokens)
        df_records['total_cost'] = cost.map(lambda v: v.cost)

        perf = df_records['perf_json'].apply(
            lambda perf_json: Perf.model_validate_json(perf_json)
            if perf_json != MIGRATION_UNKNOWN_STR else MIGRATION_UNKNOWN_STR
        )

        df_records['latency'] = perf.apply(
            lambda p: p.latency.seconds
            if p != MIGRATION_UNKNOWN_STR else MIGRATION_UNKNOWN_STR
        )

        if len(df_records) == 0:
            return df_records, []

        result_cols = set()

        def expand_results(row):
            if row['name'] is not None:

                if row.multi_result is not None and json.loads(row.multi_result
                                                              ) is not None:
                    multi_dict = json.loads(row.multi_result)
                    for output_key in multi_dict:
                        col_name = f"{row['name']}{MULTI_CALL_NAME_DELIMITER}{output_key}"
                        result_cols.add(col_name)
                        row[col_name] = multi_dict[output_key]
                else:
                    result_cols.add(row['name'])
                    row[row['name']] = row.result
                row[row['name'] + "_calls"] = json.loads(row.calls_json
                                                        )['calls']

            return pd.Series(row)

        df_results = df_results.apply(expand_results, axis=1)
        df_results = df_results.drop(
            columns=["name", "result", "multi_result", "calls_json"]
        )

        def nonempty(val):
            if isinstance(val, float):
                return not np.isnan(val)
            return True

        def merge_feedbacks(vals):
            ress = list(filter(nonempty, vals))
            if len(ress) > 0:
                return ress[0]
            else:
                return np.nan

        df_results = df_results.groupby("record_id").agg(merge_feedbacks
                                                        ).reset_index()

        assert "record_id" in df_results.columns
        assert "record_id" in df_records.columns

        combined_df = df_records.merge(df_results, on=['record_id'])

        return combined_df, list(result_cols)

    def get_feedback_count_by_status(
        self,
        record_id: Optional[RecordID] = None,
        feedback_result_id: Optional[FeedbackResultID] = None,
        feedback_definition_id: Optional[FeedbackDefinitionID] = None,
        status: Optional[Union[FeedbackResultStatus,
                               Sequence[FeedbackResultStatus]]] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        shuffle: bool = False
    ) -> Dict[FeedbackResultStatus, int]:
        
        raise NotImplementedError("This database implementation is deprecated.")
    