"""Curation of recorded traces into persisted evaluation datasets.

Production traces already contain the examples a regression test needs, but
[add_ground_truth_to_dataset][trulens.core.session.TruSession.add_ground_truth_to_dataset]
expects a dataframe that has already been extracted, joined, normalized and
deduplicated. This module does that preparation, mapping a records dataframe
onto [GroundTruth][trulens.core.schema.groundtruth.GroundTruth] rows through
the existing dataset APIs.
"""

from __future__ import annotations

import json
import logging
import math
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
)

import pandas as pd
import pydantic
from pydantic import BaseModel
from pydantic import Field
from trulens.core.schema import dataset as dataset_schema
from trulens.core.schema import groundtruth as groundtruth_schema
from trulens.core.schema import types as types_schema

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100
"""Number of ground truths written per database round trip."""

ON_ERROR_RAISE = "raise"
"""Fail on the first row that cannot be curated."""

ON_ERROR_COLLECT = "collect"
"""Skip rows that cannot be curated and report them in the result."""

ON_ERROR_MODES = (ON_ERROR_RAISE, ON_ERROR_COLLECT)

PROVENANCE_COLUMNS = {
    "source_record_id": "record_id",
    "source_app_name": "app_name",
    "source_app_version": "app_version",
}
"""Metadata keys automatically preserved from a records dataframe.

Each is copied only when the source column is present and the caller has not
already mapped that metadata key themselves.
"""


class TraceDatasetMapping(BaseModel):
    """Mapping from records dataframe columns to ground truth fields.

    Values are column names in the input dataframe, never expressions or
    callables, so a mapping is fully declarative and can be validated against
    the dataframe before anything is written.
    """

    query: str = "input"
    """Column holding the query / input of the example."""

    query_id: Optional[str] = "record_id"
    """Column holding a stable identifier for the query, if any."""

    expected_response: Optional[str] = None
    """Column holding the corrected or expected response, if any."""

    expected_chunks: Optional[str] = None
    """Column holding the expected retrieval contexts, if any."""

    metadata: Dict[str, str] = Field(default_factory=dict)
    """Metadata keys to preserve, mapped to the column they come from."""

    @pydantic.field_validator("query")
    @classmethod
    def _query_is_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("`query` must name a column.")
        return value

    def mapped_columns(self) -> List[str]:
        """Every dataframe column this mapping refers to, in a stable order."""

        columns = [self.query]
        for optional in (
            self.query_id,
            self.expected_response,
            self.expected_chunks,
        ):
            if optional is not None:
                columns.append(optional)
        columns.extend(self.metadata.values())

        seen: Set[str] = set()
        return [c for c in columns if not (c in seen or seen.add(c))]

    def missing_columns(self, dataframe: pd.DataFrame) -> List[str]:
        """The mapped columns that `dataframe` does not have."""

        available = set(dataframe.columns)
        return [c for c in self.mapped_columns() if c not in available]


class CurationError(BaseModel):
    """One input row that could not be turned into a ground truth."""

    row_index: Any = None
    """Index of the offending row in the input dataframe."""

    query_id: Optional[str] = None
    """The row's query id, when one could be read."""

    reason: str
    """Short, stable code for the kind of failure."""

    message: str
    """Human-readable detail."""


class CurationResult(BaseModel):
    """Outcome of curating a records dataframe into a dataset."""

    dataset_name: str
    """Name of the dataset written to."""

    dataset_id: types_schema.DatasetID
    """Id of the dataset written to."""

    accepted: int = 0
    """Rows that produced a distinct ground truth."""

    duplicates: int = 0
    """Rows that collapsed onto a ground truth already produced by this call.

    Ground truth ids are content-addressed, so curating the same rows again in
    a later call also writes no new rows; that idempotency is not counted here
    because it is resolved by the database rather than by this call.

    Note that a ground truth id covers its metadata too, so two records with
    the same question and answer but different provenance are two distinct
    ground truths. Pass `include_provenance=False` to deduplicate purely on
    example content."""

    rejected: int = 0
    """Rows that could not be curated."""

    ground_truth_ids: List[types_schema.GroundTruthID] = Field(
        default_factory=list
    )
    """Ids of the ground truths written, in input order, without repeats."""

    errors: List[CurationError] = Field(default_factory=list)
    """One entry per rejected row. Always empty in `raise` mode."""

    @property
    def processed(self) -> int:
        """Total number of input rows considered."""

        return self.accepted + self.duplicates + self.rejected

    def errors_df(self) -> pd.DataFrame:
        """Rejected rows as a dataframe, for inspection in a notebook."""

        return pd.DataFrame(
            data=[
                (e.row_index, e.query_id, e.reason, e.message)
                for e in self.errors
            ],
            columns=["row_index", "query_id", "reason", "message"],
        )


class CurationRowError(ValueError):
    """A single row could not be curated.

    Raised out of `curate_records_to_dataset` in `raise` mode; converted into a
    [CurationError][trulens.core.dataset.CurationError] in `collect` mode.
    """

    def __init__(self, reason: str, message: str, row_index: Any = None):
        self.reason = reason
        self.message = message
        self.row_index = row_index
        super().__init__(f"row {row_index}: {message}")


def _is_missing(value: Any) -> bool:
    """Whether a value should be treated as absent.

    Covers `None`, the NaN floats pandas produces for empty cells, and empty
    or whitespace-only strings.
    """

    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _normalize_text(value: Any) -> Optional[str]:
    """Resolve a recorded input or output into stable text.

    Records carry main inputs and outputs as strings in OTEL mode and as
    already-parsed JSON values in the legacy schema, so both have to reduce to
    the same text for the same content. Dicts and lists — including ones still
    encoded as a JSON string — are re-serialized with sorted keys so that key
    ordering does not change the content-addressed ground truth id.

    Mirrors the normalization used for run comparison in
    `trulens.core.run`.
    """

    if _is_missing(value):
        return None

    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)

    text = str(value).strip()

    if text.startswith(("{", "[")):
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed, sort_keys=True, ensure_ascii=False)

    return text


def _normalize_contexts(value: Any) -> Optional[List[Dict[str, Any]]]:
    """Normalize expected contexts into `GroundTruth.expected_chunks` shape.

    Accepts a list of dicts, a list of strings, a single string or dict, or a
    JSON encoding of any of those. Plain text becomes `{"text": ...}` so that
    every chunk is a dict, which is what `expected_chunks` is typed as.
    """

    if _is_missing(value):
        return None

    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("{", "[")):
            try:
                value = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return [{"text": text}]
        else:
            return [{"text": text}]

    if isinstance(value, dict):
        return [_json_safe(value)]

    if isinstance(value, (list, tuple)):
        chunks = []
        for element in value:
            if _is_missing(element):
                continue
            if isinstance(element, dict):
                chunks.append(_json_safe(element))
            else:
                chunks.append({"text": _normalize_text(element)})
        return chunks or None

    return [{"text": _normalize_text(value)}]


def _json_safe(value: Any) -> Any:
    """Coerce a value read out of a dataframe into something serializable.

    Pandas hands back numpy scalars and `NaT`, neither of which survives the
    json encoding that ground truth metadata goes through.
    """

    if _is_missing(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):  # numpy scalar
        try:
            return _json_safe(value.item())
        except (AttributeError, ValueError):
            pass
    return str(value)


def _metadata_of_row(
    row: pd.Series,
    mapping: TraceDatasetMapping,
    available: Set[str],
    include_provenance: bool = True,
) -> Dict[str, Any]:
    """Build the ground truth metadata for one row.

    Provenance columns are copied automatically when present so that a curated
    example can always be traced back to the record it came from; an explicit
    mapping for the same key wins.
    """

    meta: Dict[str, Any] = {}

    for key, column in PROVENANCE_COLUMNS.items() if include_provenance else ():
        if key in mapping.metadata or column not in available:
            continue
        value = _json_safe(row[column])
        if value is not None:
            meta[key] = value

    for key, column in mapping.metadata.items():
        meta[key] = _json_safe(row[column])

    return meta


def _ground_truth_of_row(
    row: pd.Series,
    row_index: Any,
    dataset_id: types_schema.DatasetID,
    mapping: TraceDatasetMapping,
    available: Set[str],
    expected_response_fn: Optional[Callable[[pd.Series], Optional[str]]],
    include_provenance: bool = True,
) -> groundtruth_schema.GroundTruth:
    """Turn one dataframe row into a ground truth.

    Raises:
        CurationRowError: If the row has no usable query, or the caller's
            callback fails.
    """

    query = _normalize_text(row[mapping.query])
    if query is None:
        raise CurationRowError(
            reason="empty_query",
            message=f"column '{mapping.query}' is empty",
            row_index=row_index,
        )

    query_id = (
        _normalize_text(row[mapping.query_id])
        if mapping.query_id is not None
        else None
    )

    expected_response = None
    if mapping.expected_response is not None:
        expected_response = _normalize_text(row[mapping.expected_response])

    # A mapped correction wins; the callback only fills in what is missing.
    if expected_response is None and expected_response_fn is not None:
        try:
            expected_response = _normalize_text(expected_response_fn(row))
        except Exception as e:
            raise CurationRowError(
                reason="expected_response_fn_failed",
                message=f"{type(e).__name__}: {e}",
                row_index=row_index,
            ) from e

    expected_chunks = None
    if mapping.expected_chunks is not None:
        try:
            expected_chunks = _normalize_contexts(row[mapping.expected_chunks])
        except Exception as e:
            raise CurationRowError(
                reason="invalid_expected_chunks",
                message=f"{type(e).__name__}: {e}",
                row_index=row_index,
            ) from e

    return groundtruth_schema.GroundTruth(
        dataset_id=dataset_id,
        query=query,
        query_id=query_id,
        expected_response=expected_response,
        expected_chunks=expected_chunks,
        meta=_metadata_of_row(row, mapping, available, include_provenance),
    )


def _batched(items: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    """Yield `items` in lists of at most `batch_size`."""

    batch: List[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _resolve_records(
    records: pd.DataFrame,
    mapping: TraceDatasetMapping,
    record_resolver: Optional[Callable[[List[str]], pd.DataFrame]],
) -> pd.DataFrame:
    """Fill in record content for a dataframe that only carries record ids.

    Resolution is attempted whenever the frame carries a `record_id` column and
    any mapped column is missing, because a missing column is equally likely to
    be record content the frame never had (a review export keyed by record id)
    or a metric score that only the database holds. Anything still missing
    afterwards is reported as a mapping error.

    Columns already present in `records` win, so corrections added by the
    caller are never overwritten by what is stored for the record.

    Raises:
        ValueError: If resolution was needed but failed.
    """

    if record_resolver is None:
        return records

    missing = mapping.missing_columns(records)
    if not missing or "record_id" not in records.columns:
        return records

    record_ids = [
        rid
        for rid in (_normalize_text(v) for v in records["record_id"])
        if rid is not None
    ]
    if not record_ids:
        return records

    logger.info(
        "Resolving %d record(s) to fill in missing column(s): %s",
        len(record_ids),
        ", ".join(missing),
    )
    try:
        resolved = record_resolver(record_ids)
    except Exception as e:
        raise ValueError(
            "The records dataframe is missing mapped column(s) "
            f"({', '.join(missing)}) and resolving them from the record ids "
            f"failed: {type(e).__name__}: {e}. Pass a dataframe that already "
            "carries the mapped columns, such as one returned by "
            "`get_records_and_feedback()`."
        ) from e

    if resolved is None or resolved.empty:
        return records

    add = [
        c
        for c in resolved.columns
        if c in missing and c not in records.columns and c != "record_id"
    ]
    if not add:
        return records

    return records.merge(
        resolved[["record_id"] + add].drop_duplicates(subset=["record_id"]),
        on="record_id",
        how="left",
    )


def curate_records_to_dataset(
    dataset_name: str,
    records: pd.DataFrame,
    db: Any,
    mapping: Optional[TraceDatasetMapping] = None,
    expected_response_fn: Optional[Callable[[pd.Series], Optional[str]]] = None,
    dataset_metadata: Optional[Dict[str, Any]] = None,
    on_error: str = ON_ERROR_RAISE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    include_provenance: bool = True,
    record_resolver: Optional[Callable[[List[str]], pd.DataFrame]] = None,
) -> CurationResult:
    """Curate a records dataframe into a persisted dataset.

    See
    [TruSession.curate_records_to_dataset][trulens.core.session.TruSession.curate_records_to_dataset]
    for the user-facing entry point and argument documentation.

    Args:
        dataset_name: Name of the dataset to write to.
        records: The rows to curate.
        db: Database to write through.
        mapping: Column mapping. Defaults to `TraceDatasetMapping()`.
        expected_response_fn: Fallback for rows with no mapped correction.
        dataset_metadata: Metadata for the dataset itself.
        on_error: `"raise"` or `"collect"`.
        batch_size: Ground truths written per database round trip.
        include_provenance: Whether to copy the source record id and app
            name/version into metadata.
        record_resolver: Given record ids, returns their records. Used only
            when `records` is missing mapped columns.

    Returns:
        A [CurationResult][trulens.core.dataset.CurationResult].

    Raises:
        ValueError: If `on_error` is not a supported mode, `batch_size` is not
            positive, or a mapped column is missing from `records`.
        CurationRowError: In `raise` mode, on the first unusable row.
    """

    if on_error not in ON_ERROR_MODES:
        raise ValueError(
            f"`on_error` must be one of {ON_ERROR_MODES}, got {on_error!r}."
        )

    if batch_size < 1:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}.")

    mapping = mapping if mapping is not None else TraceDatasetMapping()

    if not isinstance(records, pd.DataFrame):
        raise ValueError(
            "`records` must be a pandas DataFrame, got "
            f"{type(records).__name__}."
        )

    records = _resolve_records(records, mapping, record_resolver)

    # Validate before any write, so a bad mapping never leaves a partially
    # curated dataset behind.
    missing = mapping.missing_columns(records)
    if missing:
        raise ValueError(
            "The records dataframe is missing mapped column(s): "
            + ", ".join(missing)
            + ". Available columns: "
            + ", ".join(str(c) for c in records.columns)
            + "."
        )

    dataset_id = db.insert_dataset(
        dataset=dataset_schema.Dataset(name=dataset_name, meta=dataset_metadata)
    )

    result = CurationResult(dataset_name=dataset_name, dataset_id=dataset_id)
    available = set(records.columns)
    seen: Set[types_schema.GroundTruthID] = set()

    def _curated() -> Iterable[groundtruth_schema.GroundTruth]:
        """Yield the distinct ground truths of `records`, lazily.

        Generating lazily is what keeps peak memory to one batch of ground
        truths rather than one per input row.
        """

        for row_index, row in records.iterrows():
            try:
                ground_truth = _ground_truth_of_row(
                    row=row,
                    row_index=row_index,
                    dataset_id=dataset_id,
                    mapping=mapping,
                    available=available,
                    expected_response_fn=expected_response_fn,
                    include_provenance=include_provenance,
                )
            except CurationRowError as e:
                if on_error == ON_ERROR_RAISE:
                    raise
                result.rejected += 1
                result.errors.append(
                    CurationError(
                        row_index=_json_safe(row_index),
                        query_id=(
                            _normalize_text(row[mapping.query_id])
                            if mapping.query_id is not None
                            else None
                        ),
                        reason=e.reason,
                        message=e.message,
                    )
                )
                continue

            if ground_truth.ground_truth_id in seen:
                result.duplicates += 1
                continue

            seen.add(ground_truth.ground_truth_id)
            result.accepted += 1
            result.ground_truth_ids.append(ground_truth.ground_truth_id)
            yield ground_truth

    for batch in _batched(_curated(), batch_size):
        db.batch_insert_ground_truth(batch)

    logger.info(
        "Curated %d row(s) into dataset '%s': %d accepted, %d duplicate, "
        "%d rejected.",
        result.processed,
        dataset_name,
        result.accepted,
        result.duplicates,
        result.rejected,
    )

    return result
