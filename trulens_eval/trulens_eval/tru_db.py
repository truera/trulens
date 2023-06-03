import abc
from datetime import datetime
import json
import logging
from pathlib import Path
from pprint import PrettyPrinter
import sqlite3
from typing import (Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple,
                    Union)

from frozendict import frozendict
from merkle_json import MerkleJson
import pandas as pd
import pydantic

from trulens_eval.schema import ChainID
from trulens_eval.schema import FeedbackDefinition
from trulens_eval.schema import FeedbackDefinitionID
from trulens_eval.schema import FeedbackResult
from trulens_eval.schema import FeedbackResultID
from trulens_eval.schema import JSONPath
from trulens_eval.schema import Model
from trulens_eval.schema import Record
from trulens_eval.schema import RecordChainCall
from trulens_eval.schema import RecordID
from trulens_eval.schema import Cost
from trulens_eval.schema import FeedbackResultStatus
from trulens_eval.util import GetItemOrAttribute
from trulens_eval.util import _project
from trulens_eval.util import all_queries
from trulens_eval.util import GetItem
from trulens_eval.util import JSON
from trulens_eval.util import JSON_BASES
from trulens_eval.util import json_str_of_obj
from trulens_eval.util import JSONPath
from trulens_eval.util import noserio
from trulens_eval.util import obj_id_of_obj
from trulens_eval.util import SerialModel
from trulens_eval.util import UNCIODE_YIELD
from trulens_eval.util import UNICODE_CHECK

mj = MerkleJson()
NoneType = type(None)

pp = PrettyPrinter()


class Query:

    # Typing for type hints.
    Query = JSONPath

    # Instance for constructing queries for record json like `Record.chain.llm`.
    Record = Query().__record__

    # Instance for constructing queries for chain json.
    Chain = Query().__chain__

    # A Chain's main input and main output.
    # TODO: Chain input/output generalization.
    RecordInput = Record.main_input
    RecordOutput = Record.main_output


def get_calls(record: Record) -> Iterable[RecordChainCall]:
    """
    Iterate over the call parts of the record.
    """

    for q in all_queries(record):
        print("consider query", q)
        if len(q.path) > 0 and q.path[-1] == GetItemOrAttribute(item_or_attribute="_call"):
            yield q


def get_calls_by_stack(
    record: Record
) -> Dict[Tuple[str, ...], RecordChainCall]:
    """
    Get a dictionary mapping chain call stack to the call information.
    """

    def frozen_frame(frame):
        frame['path'] = tuple(frame['path'])
        return frozendict(frame)

    ret = dict()

    for c in get_calls(record):
        print("call", c)

        obj = TruDB.project(c, record_json=record, chain_json=None)
        if isinstance(obj, Sequence):
            for o in obj:
                call_stack = tuple(map(frozen_frame, o['chain_stack']))
                if call_stack not in ret:
                    ret[call_stack] = []
                ret[call_stack].append(o)
        else:
            call_stack = tuple(map(frozen_frame, obj['chain_stack']))
            if call_stack not in ret:
                ret[call_stack] = []
            ret[call_stack].append(obj)

    return ret


"""
def query_of_path(path: List[Union[str, int]]) -> JSONPath:

    if path[0] == "_record":
        ret = Query.Record
        path = path[1:]
    elif path[0] == "_chain":
        ret = Record.Chain
        path = path[1:]
    else:
        ret = JSONPath()

    for attr in path:
        ret = getattr(ret, attr)

    return ret
"""


class TruDB(SerialModel, abc.ABC):
    @abc.abstractmethod
    def reset_database(self):
        """
        Delete all data.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_record(
        self,
        record: Record,
    ) -> RecordID:
        """
        Insert a new `record` into db, indicating its `model` as well. Return
        record id.

        Args:
        - record: Record
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_chain(self, chain: Model) -> ChainID:
        """
        Insert a new `chain` into db under the given `chain_id`. 

        Args:
        - chain: Chain - Chain definition. 
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_feedback_definition(
        self, feedback_definition: FeedbackDefinition
    ) -> FeedbackDefinitionID:
        """
        Insert a feedback definition into the db.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_feedback(
        self,
        feedback_result: FeedbackResult,
    ) -> FeedbackResultID:
        """
        Insert a feedback record into the db.

        Args:

        - feedback_result: FeedbackResult
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def get_records_and_feedback(
        self, chain_ids: List[str]
    ) -> Tuple[pd.DataFrame, Sequence[str]]:
        """
        Get the records logged for the given set of `chain_ids` (otherwise all)
        alongside the names of the feedback function columns listed the
        dataframe.
        """
        raise NotImplementedError()


class LocalSQLite(TruDB):
    filename: Path

    TABLE_RECORDS = "records"
    TABLE_FEEDBACKS = "feedbacks"
    TABLE_FEEDBACK_DEFS = "feedback_defs"
    TABLE_CHAINS = "chains"

    TYPE_TIMESTAMP = "FLOAT"
    TYPE_ENUM = "TEXT"

    TABLES = [TABLE_RECORDS, TABLE_FEEDBACKS, TABLE_FEEDBACK_DEFS, TABLE_CHAINS]

    def __init__(self, filename: Path):
        """
        Database locally hosted using SQLite.

        Args
        
        - filename: Optional[Path] -- location of sqlite database dump
          file. It will be created if it does not exist.

        """
        super().__init__(filename=filename)

        self._build_tables()

    def __str__(self) -> str:
        return f"SQLite({self.filename})"

    # TruDB requirement
    def reset_database(self) -> None:
        self._drop_tables()
        self._build_tables()

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

    def _build_tables(self):
        conn, c = self._connect()

        # Create table if it does not exist. Note that the record_json column
        # also encodes inside it all other columns.
        c.execute(
            f'''CREATE TABLE IF NOT EXISTS {self.TABLE_RECORDS} (
                record_id TEXT NOT NULL PRIMARY KEY,
                chain_id TEXT NOT NULL,
                input TEXT,
                output TEXT,
                record_json TEXT NOT NULL,
                tags TEXT NOT NULL,
                ts {self.TYPE_TIMESTAMP} NOT NULL,
                cost_json TEXT NOT NULL
            )'''
        )
        c.execute(
            f'''CREATE TABLE IF NOT EXISTS {self.TABLE_FEEDBACKS} (
                record_id TEXT NOT NULL,
                chain_id TEXT NOT NULL,
                feedback_result_id TEXT NOT NULL PRIMARY KEY,
                feedback_definition_id TEXT,
                last_ts {self.TYPE_TIMESTAMP} NOT NULL,
                status {self.TYPE_ENUM} NOT NULL,
                error TEXT,
                results_json TEXT NOT NULL,
                cost_json TEXT NOT NULL
            )'''
        )
        c.execute(
            f'''CREATE TABLE IF NOT EXISTS {self.TABLE_FEEDBACK_DEFS} (
                feedback_definition_id TEXT NOT NULL PRIMARY KEY,
                feedback_json TEXT NOT NULL
            )'''
        )
        c.execute(
            f'''CREATE TABLE IF NOT EXISTS {self.TABLE_CHAINS} (
                chain_id TEXT NOT NULL PRIMARY KEY,
                chain_json TEXT NOT NULL
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

    # TruDB requirement
    def insert_record(
        self,
        record: Record,
    ) -> RecordID:
        # NOTE: Oddness here in that the entire record is put into the
        # record_json column while some parts of that records are also put in
        # other columns. Might want to keep this so we can query on the columns
        # within sqlite.

        conn, c = self._connect()
        record_json_str = json_str_of_obj(record)
        cost_json_str = json_str_of_obj(record.cost)

        c.execute(
            f"INSERT INTO {self.TABLE_RECORDS} VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.record_id, record.chain_id, record.main_input,
                record.main_output, record_json_str, record.tags, record.ts,
                cost_json_str
            )
        )
        self._close(conn)

        print(
            f"{UNICODE_CHECK} record {record.record_id} from {record.chain_id} -> {self.filename}"
        )

        return record.record_id

    # TruDB requirement
    def insert_chain(self, chain: Model) -> ChainID:
        chain_id = chain.chain_id

        chain_str = chain.json()

        conn, c = self._connect()
        c.execute(
            f"INSERT OR IGNORE INTO {self.TABLE_CHAINS} VALUES (?, ?)",
            (chain_id, chain_str)
        )
        self._close(conn)

        print(f"{UNICODE_CHECK} chain {chain_id} -> {self.filename}")

        return chain_id

    def insert_feedback_definition(
        self, feedback: FeedbackDefinition
    ) -> FeedbackDefinitionID:
        """
        Insert a feedback definition into the database.
        """

        feedback_definition_id = feedback.feedback_definition_id
        feedback_str = feedback.json()

        conn, c = self._connect()
        c.execute(
            f"INSERT OR REPLACE INTO {self.TABLE_FEEDBACK_DEFS} VALUES (?, ?)",
            (feedback_definition_id, feedback_str)
        )
        self._close(conn)

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

    def insert_feedback(
        self, feedback_result: FeedbackResult
    ) -> FeedbackResultID:
        """
        Insert a record-feedback link to db or update an existing one.
        """

        feedback_results_json_str = json_str_of_obj(feedback_result.results_json)
        cost_json_str = json_str_of_obj(feedback_result.cost)

        conn, c = self._connect()
        c.execute(
            f"""INSERT OR REPLACE INTO {self.TABLE_FEEDBACKS}
                VALUES (?, ?, ?, ?, ?,
                        ?, ?, ?, ?)""", (
                feedback_result.record_id,
                feedback_result.chain_id, 
                feedback_result.feedback_result_id,
                feedback_result.feedback_definition_id, 
                feedback_result.last_ts.timestamp(),
                feedback_result.status.value, feedback_result.error,
                feedback_results_json_str, cost_json_str
            )
        )
        self._close(conn)

        if feedback_result.status == 2:
            print(
                f"{UNICODE_CHECK} feedback {feedback_result.feedback_result_id} on {feedback_result.record_id} -> {self.filename}"
            )
        else:
            print(
                f"{UNCIODE_YIELD} feedback {feedback_result.feedback_result_id} on {feedback_result.record_id} -> {self.filename}"
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
            clauses.append("record_id=?")
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
                f.last_ts, f.status, f.error,
                f.results_json, 
                f.cost_json,
                fd.feedback_json, 
                r.record_json, 
                c.chain_json
            FROM {self.TABLE_RECORDS} r
                JOIN {self.TABLE_FEEDBACKS} f 
                JOIN {self.TABLE_FEEDBACK_DEFS} fd
                JOIN {self.TABLE_CHAINS} c
            WHERE f.feedback_definition_id=fd.feedback_definition_id
                AND r.record_id=f.record_id
                AND r.chain_id=c.chain_id
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
            
            row.results_json = json.loads(row.results_json)  # results_json (unstructured)
            row.cost_json = json.loads(row.cost_json)  # cost_json (Cost)
            row.feedback_json = json.loads(row.feedback_json)  # feedback_json (FeedbackDefinition)
            row.record_json = json.loads(row.record_json)  # record_json (Record)
            row.chain_json = json.loads(row.chain_json)  # chain_json (Model)

            row.status = FeedbackResultStatus(row.status)

            row['total_tokens'] = row.cost_json['n_tokens']
            row['total_cost'] = row.cost_json['cost']

            return row

        df = df.apply(map_row, axis=1)

        return pd.DataFrame(df)

    def get_chain(self, chain_id: str) -> JSON:
        conn, c = self._connect()
        c.execute(
            f"SELECT chain_json FROM {self.TABLE_CHAINS} WHERE chain_id=?",
            (chain_id,)
        )
        result = c.fetchone()[0]
        conn.close()

        return json.loads(result)

    def get_records_and_feedback(
        self, chain_ids: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Sequence[str]]:
        # This returns all models if the list of chain_ids is empty.
        chain_ids = chain_ids or []

        conn, c = self._connect()
        query = f"""
            SELECT r.record_id, f.results_json
            FROM {self.TABLE_RECORDS} r 
            LEFT JOIN {self.TABLE_FEEDBACKS} f
                ON r.record_id = f.record_id
            """
        if len(chain_ids) > 0:
            chain_id_list = ', '.join('?' * len(chain_ids))
            query = query + f" WHERE r.chain_id IN ({chain_id_list})"

        c.execute(query)
        rows = c.fetchall()
        conn.close()

        df_results = pd.DataFrame(
            rows, columns=[description[0] for description in c.description]
        )

        if len(df_results) == 0:
            return df_results, []

        conn, c = self._connect()
        query = f"""
            SELECT DISTINCT r.*, c.chain_json
            FROM {self.TABLE_RECORDS} r 
            JOIN {self.TABLE_CHAINS} c
                ON r.chain_id = c.chain_id
            """
        if len(chain_ids) > 0:
            chain_id_list = ', '.join('?' * len(chain_ids))
            query = query + f" WHERE r.chain_id IN ({chain_id_list})"

        c.execute(query)
        rows = c.fetchall()
        conn.close()

        df_records = pd.DataFrame(
            rows, columns=[description[0] for description in c.description]
        )

        cost = df_records['cost_json'].map(Cost.parse_raw)
        df_records['total_tokens'] = cost.map(lambda v: v.n_tokens)
        df_records['total_cost'] = cost.map(lambda v: v.cost)

        if len(df_records) == 0:
            return df_records, []

        # Apply the function to the 'data' column to convert it into separate columns
        df_results['results_json'] = df_results['results_json'].apply(
            json.loads
        )

        if "record_id" not in df_results.columns:
            return df_results, []

        df_results = df_results.groupby("record_id").agg(
            lambda dicts: {
                key: val for d in dicts if d is not None
                for key, val in d.items()
            }
        ).reset_index()

        def expand_results(row):
            s = pd.Series(row.results_json)
            s['record_id'] = row.record_id
            return s

        df_results = df_results.apply(expand_results, axis=1)

        result_cols = [
            col for col in df_results.columns
            if col not in ['feedback_id', 'record_id', '_success', "_error"]
        ]

        if len(df_results) == 0 or len(result_cols) == 0:
            return df_records, []

        assert "record_id" in df_results.columns
        assert "record_id" in df_records.columns

        combined_df = df_records.merge(df_results, on=['record_id'])
        combined_df = combined_df.drop(
            columns=set(["_success", "_error", 'cost_json']
                       ).intersection(set(combined_df.columns))
        )

        return combined_df, result_cols
