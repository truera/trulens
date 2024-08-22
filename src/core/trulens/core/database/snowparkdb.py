from datetime import datetime
from typing import Dict, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col
from trulens.core.database.base import DB
import logging
from trulens.core.utils.python import locals_except
from trulens.core.schema import AppDefinition
from trulens.core.schema.types import FeedbackDefinitionID
from trulens.core.schema import types as mod_types_schema
from trulens.core.schema import feedback as mod_feedback_schema
from trulens.core.database import orm as mod_orm
from trulens.core.schema import base as mod_base_schema
import json
import pandas as pd

logger = logging.getLogger(__name__)

class SnowparkDB(DB):
    session: Session = None
    redact_keys: list = []
    orm: Type[mod_orm.ORM]
    class Config:
        arbitrary_types_allowed = True
    def __init__(self, session: Session, redact_keys):
        self.session = session
        self.redact_keys = redact_keys
        self.orm = mod_orm.make_orm_for_prefix()

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
    
    def _feedback_query(
    self,
    count_by_status: bool = False,
    shuffle: bool = False,
    record_id: Optional[mod_types_schema.RecordID] = None,
    feedback_result_id: Optional[mod_types_schema.FeedbackResultID] = None,
    feedback_definition_id: Optional[
        mod_types_schema.FeedbackDefinitionID
    ] = None,
        status: Optional[
            Union[
                mod_feedback_schema.FeedbackResultStatus,
                Sequence[mod_feedback_schema.FeedbackResultStatus],
            ]
        ] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        run_location: Optional[mod_feedback_schema.FeedbackRunLocation] = None,
    ):
        if count_by_status:
            q = self.session.table(self.orm.FeedbackResult.__tablename__).group_by('status').agg(
                self.session.functions.count('feedback_result_id').alias('count')
            )
        else:
            q = self.session.table(self.orm.FeedbackResult.__tablename__)

        if record_id:
            q = q.filter(q['record_id'] == record_id)

        if feedback_result_id:
            q = q.filter(q['feedback_result_id'] == feedback_result_id)

        if feedback_definition_id:
            q = q.filter(q['feedback_definition_id'] == feedback_definition_id)

        # if run_location is None or run_location == mod_feedback_schema.FeedbackRunLocation.IN_APP:
        #     q = q.filter(
        #         (q['run_location'].is_null()) |
        #         (q['run_location'] == mod_feedback_schema.FeedbackRunLocation.IN_APP.value)
        #     )
        # else:
        #     q = q.filter(q['run_location'] == run_location.value)

        r = self.session.table(self.orm.FeedbackDefinition.__tablename__)
        f = self.session.table(self.orm.FeedbackDefinition.__tablename__)
        r = self.session.table(self.orm.Record.__tablename__)
        a = self.session.table(self.orm.AppDefinition.__tablename__)
        q = q.join(
                f, q['feedback_definition_id'] == f['feedback_definition_id']
            ).join(
                r, q['record_id'] == r['record_id']
            ).join(
                a, r['app_id'] == a['app_id']
            ).select(
                q['record_id'].alias("record_id"),
                q['feedback_result_id'],
                q['feedback_definition_id'].alias("feedback_definition_id"),
                q['last_ts'],
                q['status'],
                q['error'],
                q['name'].alias("fname"),
                q['result'],
                q['multi_result'],
                q['cost_json'].alias("cost_json"),
                r['perf_json'],
                q['calls_json'],
                f['feedback_json'],
                r['record_json'],
                a['app_json'],
                # q['type'],
            )

        if status:
            if isinstance(status, mod_feedback_schema.FeedbackResultStatus):
                status = [status.value]
            q = q.filter(q['status'].isin(status))

        if last_ts_before:
            q = q.filter(q['last_ts'] < last_ts_before.timestamp())

        if offset is not None:
            q = q.offset(offset)

        if limit is not None:
            q = q.limit(limit)

        if shuffle:
            q = q.order_by(self.session.functions.random())

        return q

    def get_feedback_count_by_status(
        self,
        record_id: Optional[mod_types_schema.RecordID] = None,
        feedback_result_id: Optional[mod_types_schema.FeedbackResultID] = None,
        feedback_definition_id: Optional[
            mod_types_schema.FeedbackDefinitionID
        ] = None,
        status: Optional[
            Union[
                mod_feedback_schema.FeedbackResultStatus,
                Sequence[mod_feedback_schema.FeedbackResultStatus],
            ]
        ] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        shuffle: bool = False,
        run_location: Optional[mod_feedback_schema.FeedbackRunLocation] = None,
    ) -> Dict[mod_feedback_schema.FeedbackResultStatus, int]:
        """See [DB.get_feedback_count_by_status][trulens.core.database.base.DB.get_feedback_count_by_status]."""

        q = self._feedback_query(
            count_by_status=True,
            **locals_except("self", "session"),
        )
        results = q.collect()

        return {
            mod_feedback_schema.FeedbackResultStatus(row[0]): row[1]
            for row in results
        }

    def get_feedback(
        self,
        record_id: Optional[mod_types_schema.RecordID] = None,
        feedback_result_id: Optional[mod_types_schema.FeedbackResultID] = None,
        feedback_definition_id: Optional[
            mod_types_schema.FeedbackDefinitionID
        ] = None,
        status: Optional[
            Union[
                mod_feedback_schema.FeedbackResultStatus,
                Sequence[mod_feedback_schema.FeedbackResultStatus],
            ]
        ] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        shuffle: Optional[bool] = False,
        run_location: Optional[mod_feedback_schema.FeedbackRunLocation] = None,
    ) -> pd.DataFrame:
        """See [DB.get_feedback][trulens.core.database.base.DB.get_feedback]."""

        q = self._feedback_query(**locals_except("self"))

        return q.to_pandas()

    def get_records_and_feedback(
        self,
        app_ids: Optional[List[str]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Sequence[str]]:
        """See [DB.get_records_and_feedback][trulens.core.database.base.DB.get_records_and_feedback]."""

        # TODO: Add pagination to this method. Currently the joinedload in
        # select below disables lazy loading of records which will be a problem
        # for large databases without the use of pagination.

        with self.session.begin() as session:
            stmt = sa.select(self.orm.Record)
            # NOTE: We are selecting records here because offset and limit need
            # to be with respect to those rows instead of AppDefinition or
            # FeedbackResult rows.

            if app_ids:
                stmt = stmt.where(self.orm.Record.app_id.in_(app_ids))

            stmt = stmt.options(joinedload(self.orm.Record.feedback_results))
            # NOTE(piotrm): The joinedload here makes it so that the
            # feedback_results get loaded eagerly instead if lazily when
            # accessed later.

            # TODO(piotrm): The subsequent logic in helper methods end up
            # reading all of the records and feedback_results in order to create
            # a DataFrame so there is no reason to not eagerly get all of this
            # data. Ideally, though, we would be making some sort of lazy
            # DataFrame and then could use the lazy join feature of sqlalchemy.

            stmt = stmt.order_by(self.orm.Record.ts, self.orm.Record.record_id)
            # NOTE: feedback_results order is governed by the order_by on the
            # orm.FeedbackResult.record backref definition. Here, we need to
            # order Records as we did not use an auto join to retrieve them. If
            # records were to be retrieved from AppDefinition.records via auto
            # join, though, the orm backref ordering would be able to take hold.

            stmt = stmt.limit(limit).offset(offset)

            ex = session.execute(stmt).unique()
            # unique needed for joinedload above.

            records = [rec[0] for rec in ex]
            # TODO: Make the iteration of records lazy in some way. See
            # TODO(piotrm) above.

            return AppsExtractor().get_df_and_cols(records=records)

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
    
no_perf = mod_base_schema.Perf.min().model_dump()
def _extract_feedback_results(
    results: Iterable["mod_orm.FeedbackResult"],
) -> pd.DataFrame:
    def _extract(_result: "mod_orm.FeedbackResult"):
        app_json = json.loads(_result.record.app.app_json)
        _type = AppDefinition.model_validate(app_json).root_class

        return (
            _result.record_id,
            _result.feedback_result_id,
            _result.feedback_definition_id,
            _result.last_ts,
            mod_feedback_schema.FeedbackResultStatus(_result.status),
            _result.error,
            _result.name,
            _result.result,
            _result.multi_result,
            _result.cost_json,  # why is cost_json not parsed?
            json.loads(_result.record.perf_json)
            if _result.record.perf_json != "unknown[db_migration]"
            else no_perf,
            json.loads(_result.calls_json)["calls"],
            json.loads(_result.feedback_definition.feedback_json)
            if _result.feedback_definition is not None
            else None,
            json.loads(_result.record.record_json),
            app_json,
            _type,
        )

    df = pd.DataFrame(
        data=(_extract(r) for r in results),
        columns=[
            "record_id",
            "feedback_result_id",
            "feedback_definition_id",
            "last_ts",
            "status",
            "error",
            "fname",
            "result",
            "multi_result",
            "cost_json",
            "perf_json",
            "calls_json",
            "feedback_json",
            "record_json",
            "app_json",
            "type",
        ],
    )
    df["latency"] = _extract_latency(df["perf_json"])
    df = pd.concat([df, _extract_tokens_and_cost(df["cost_json"])], axis=1)
    return df


def _extract_latency(
    series: Iterable[Union[str, dict, mod_base_schema.Perf]],
) -> pd.Series:
    def _extract(perf_json: Union[str, dict, mod_base_schema.Perf]) -> int:
        if perf_json == "unknown[db_migration]":
            return np.nan

        if isinstance(perf_json, str):
            perf_json = json.loads(perf_json)

        if isinstance(perf_json, dict):
            perf_json = mod_base_schema.Perf.model_validate(perf_json)

        if isinstance(perf_json, mod_base_schema.Perf):
            return perf_json.latency.seconds

        if perf_json is None:
            return 0

        raise ValueError(f"Failed to parse perf_json: {perf_json}")

    return pd.Series(data=(_extract(p) for p in series))


def _extract_tokens_and_cost(cost_json: pd.Series) -> pd.DataFrame:
    def _extract(_cost_json: Union[str, dict]) -> Tuple[int, float]:
        if isinstance(_cost_json, str):
            _cost_json = json.loads(_cost_json)
        if _cost_json is not None:
            cost = mod_base_schema.Cost(**_cost_json)
        else:
            cost = mod_base_schema.Cost()
        return cost.n_tokens, cost.cost

    return pd.DataFrame(
        data=(_extract(c) for c in cost_json),
        columns=["total_tokens", "total_cost"],
    )


def _extract_ground_truths(
    results: Iterable["mod_orm.GroundTruth"],
) -> pd.DataFrame:
    def _extract(_result: "mod_orm.GroundTruth"):
        ground_truth_json = json.loads(_result.ground_truth_json)

        return (
            _result.ground_truth_id,
            _result.dataset_id,
            ground_truth_json["query"],
            ground_truth_json["query_id"],
            ground_truth_json["expected_response"],
            ground_truth_json["expected_chunks"],
            ground_truth_json["meta"],
        )

    return pd.DataFrame(
        data=(_extract(r) for r in results),
        columns=[
            "ground_truth_id",
            "dataset_id",
            "query",
            "query_id",
            "expected_response",
            "expected_chunks",
            "meta",
        ],
    )