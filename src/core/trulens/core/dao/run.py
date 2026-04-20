from __future__ import annotations

import abc
import logging
from typing import Any, List, Optional

import pandas as pd
from trulens.core.enums import Mode

logger = logging.getLogger(__name__)


class RunDaoBase(abc.ABC):
    """Abstract base class for Run data access objects.

    Defines the interface that both the OSS (SQLAlchemy-backed) and
    Snowflake implementations must satisfy so that ``Run`` and ``App``
    can work identically regardless of backend.
    """

    @abc.abstractmethod
    def create_new_run(
        self,
        object_name: str,
        object_type: str,
        run_name: str,
        dataset_name: str,
        source_type: str,
        dataset_spec: dict,
        object_version: Optional[str] = None,
        description: Optional[str] = "",
        label: Optional[str] = "",
        llm_judge_name: Optional[str] = "",
        mode: Optional[Mode] = Mode.APP_INVOCATION,
    ) -> pd.DataFrame: ...

    @abc.abstractmethod
    def get_run(
        self,
        run_name: str,
        object_name: str,
        object_type: str,
        object_version: Optional[str] = None,
    ) -> pd.DataFrame: ...

    @abc.abstractmethod
    def list_all_runs(
        self,
        object_name: str,
        object_type: str,
    ) -> pd.DataFrame: ...

    @abc.abstractmethod
    def delete_run(
        self,
        run_name: str,
        object_name: str,
        object_type: str,
        object_version: Optional[str] = None,
    ) -> None: ...

    @abc.abstractmethod
    def upsert_run_metadata_fields(
        self,
        run_name: str,
        object_name: str,
        object_type: str,
        object_version: Optional[str] = None,
        entry_id: Optional[str] = None,
        entry_type: Optional[str] = None,
        **field_updates: Any,
    ) -> None: ...

    @abc.abstractmethod
    def start_ingestion_query(
        self,
        object_name: str,
        object_version: Optional[str],
        object_type: str,
        run_name: str,
        input_records_count: int,
    ) -> None: ...

    @abc.abstractmethod
    def call_compute_metrics_query(
        self,
        metrics: List[str],
        object_name: str,
        object_version: Optional[str],
        object_type: str,
        run_name: str,
    ) -> None: ...

    def fetch_source_data(self, source_name: str) -> pd.DataFrame:
        raise NotImplementedError(
            "Source table fetch not supported by this DAO."
        )
