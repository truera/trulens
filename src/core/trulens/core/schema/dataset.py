"""Serializable dataset-related classes."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Hashable, List, Optional, Sequence

import pydantic
from trulens.core.schema import types as types_schema
from trulens.core.utils import json as json_utils
from trulens.core.utils import serial as serial_utils

logger = logging.getLogger(__name__)

DATASET_VERSION_ID_PREFIX = "dataset_version_hash_"
"""Prefix given to every content-addressed dataset version id."""

DATASET_VERSION_ITEM_ID_PREFIX = "dataset_version_item_hash_"
"""Prefix given to every content-addressed dataset version item id."""

VERSION_ZERO_CREATED_AT = 0.0
"""Creation timestamp reported for version zero.

Version zero is reconstructed from
[GroundTruth][trulens.core.schema.groundtruth.GroundTruth] rows that predate
versioning and therefore carry no trustworthy creation time. A fixed value
keeps the reconstruction deterministic.
"""

VERSION_ZERO_ORIGIN = "ground_truth_compatibility"
"""Value of the `origin` key in version zero's source metadata."""


def _is_missing(value: Any) -> bool:
    """Whether a value should be normalized away to `None`.

    Covers `None` as well as the NaN floats that pandas produces for empty
    cells, which would otherwise make otherwise-identical examples hash
    differently.
    """

    if value is None:
        return True
    # NaN is the only value that is not equal to itself.
    return isinstance(value, float) and value != value


def _normalize_content(value: Any) -> Any:
    """Normalize example content so cosmetic differences do not change ids.

    Strings are stripped, missing values collapse to `None`, and containers
    are normalized element-wise while preserving their order.
    """

    if _is_missing(value):
        return None
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return [_normalize_content(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_content(v) for k, v in value.items()}
    return value


def _canonical_json(value: Any) -> str:
    """Serialize a value so equal content always yields equal bytes.

    Dictionary keys are sorted but list order is preserved, which is what
    makes a version id depend on the *order* of its items.
    """

    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _content_hash(value: Any) -> str:
    """Hash a json-able structure into a hex digest."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


COLUMN_SPEC_FIELDS = (
    "input",
    "input_id",
    "expected_response",
    "expected_contexts",
    "metadata",
)
"""Canonical column specification keys understood by dataset versions."""

COLUMN_SPEC_ALIASES = {
    "query": "input",
    "record_root.input": "input",
    "query_id": "input_id",
    "ground_truth_output": "expected_response",
    "record_root.ground_truth_output": "expected_response",
    "expected_chunks": "expected_contexts",
    "meta": "metadata",
}
"""Accepted spellings for the canonical column specification keys.

These cover both the legacy
[GroundTruth][trulens.core.schema.groundtruth.GroundTruth] field names and the
reserved `dataset_spec` keys used by
[RunConfig][trulens.core.run.RunConfig], so the same mapping can be handed to
both APIs.
"""


def normalize_column_spec(column_spec: Dict[str, str]) -> Dict[str, str]:
    """Resolve a user column specification to canonical item fields.

    Keys are matched case-insensitively and through
    [COLUMN_SPEC_ALIASES][trulens.core.schema.dataset.COLUMN_SPEC_ALIASES].
    Keys that do not name a dataset version item field are dropped, which lets
    a `RunConfig.dataset_spec` carrying extra span attributes be reused as-is.

    Args:
        column_spec: Mapping from a field name to a dataframe column name.

    Returns:
        A mapping keyed by
        [COLUMN_SPEC_FIELDS][trulens.core.schema.dataset.COLUMN_SPEC_FIELDS].

    Raises:
        ValueError: If no `input` column can be resolved.
    """

    normalized: Dict[str, str] = {}
    ignored = []

    for key, value in (column_spec or {}).items():
        lowered = str(key).lower()
        field = COLUMN_SPEC_ALIASES.get(lowered, lowered)
        if field in COLUMN_SPEC_FIELDS:
            normalized[field] = value
        else:
            ignored.append(key)

    if ignored:
        # Warn rather than debug: a caller passing something like
        # {"retrieval.query_text": "question"} expected those columns to be
        # read, and a silent drop only surfaces later as missing item data.
        logger.warning(
            "Ignoring column spec entries that do not map to dataset version "
            "item fields: %s. Supported fields: %s.",
            ", ".join(sorted(str(k) for k in ignored)),
            ", ".join(sorted(COLUMN_SPEC_FIELDS)),
        )

    if "input" not in normalized:
        raise ValueError(
            "The column spec must map an `input` column. Supported keys are: "
            + ", ".join(COLUMN_SPEC_FIELDS)
            + "."
        )

    return normalized


class Dataset(serial_utils.SerialModel, Hashable):
    """The class that holds the metadata of a dataset stored in the DB."""

    dataset_id: types_schema.DatasetID  # str
    """The unique identifier for the dataset."""

    name: str
    """The name of the dataset."""

    meta: types_schema.Metadata  # dict
    """Metadata associated with the dataset."""

    def __init__(
        self,
        name: str,
        dataset_id: Optional[types_schema.DatasetID] = None,
        meta: Optional[types_schema.Metadata] = None,
        **kwargs,
    ):
        kwargs["name"] = name
        kwargs["meta"] = meta if meta is not None else {}
        super().__init__(
            dataset_id="temporary", **kwargs
        )  # dataset_id will be updated below

        if dataset_id is None:
            dataset_id = json_utils.obj_id_of_obj(
                json_utils.jsonify(self), prefix="dataset"
            )

        self.dataset_id = dataset_id

    def __hash__(self):
        return hash(self.dataset_id)


class DatasetVersionItem(serial_utils.SerialModel, Hashable):
    """A single example belonging to an immutable dataset version.

    The `item_id` is content-addressed from the normalized example content
    (`input`, `expected_response`, `expected_contexts`) together with the
    optional caller-supplied `input_id`. `meta` and `splits` are properties of
    the item *within a version* and deliberately do not take part in item
    identity, so annotating an example does not make membership comparison
    report it as removed and re-added.
    """

    item_id: types_schema.DatasetVersionItemID  # str
    """The content-addressed identifier for this example."""

    dataset_version_id: Optional[types_schema.DatasetVersionID] = None
    """The version this item belongs to.

    `None` until the item is attached to a published version."""

    input_id: Optional[str] = None
    """Caller-supplied stable identifier for the example, if any.

    When given it participates in `item_id`, which lets two examples with
    identical content stay distinct."""

    input: Optional[str] = None
    """The query / input of the example."""

    expected_response: Optional[str] = None
    """The expected response for the input."""

    expected_contexts: Optional[List[Any]] = None
    """The expected retrieval contexts for the input."""

    meta: types_schema.Metadata = pydantic.Field(default_factory=dict)
    """Metadata for the example within this version."""

    splits: List[str] = pydantic.Field(default_factory=list)
    """Names of the splits this example belongs to within this version."""

    def __init__(
        self,
        input: Optional[str] = None,
        input_id: Optional[str] = None,
        expected_response: Optional[str] = None,
        expected_contexts: Optional[Sequence[Any]] = None,
        meta: Optional[types_schema.Metadata] = None,
        splits: Optional[Sequence[str]] = None,
        item_id: Optional[types_schema.DatasetVersionItemID] = None,
        dataset_version_id: Optional[types_schema.DatasetVersionID] = None,
        **kwargs,
    ):
        normalized_contexts = _normalize_content(expected_contexts)

        kwargs["input"] = _normalize_content(input)
        kwargs["input_id"] = _normalize_content(input_id)
        kwargs["expected_response"] = _normalize_content(expected_response)
        kwargs["expected_contexts"] = normalized_contexts
        kwargs["meta"] = _normalize_content(meta) if meta is not None else {}
        kwargs["splits"] = sorted({str(s) for s in (splits or [])})
        kwargs["dataset_version_id"] = dataset_version_id

        super().__init__(item_id="temporary", **kwargs)  # updated below

        self.item_id = item_id if item_id is not None else self.compute_id()

    def compute_id(self) -> types_schema.DatasetVersionItemID:
        """Compute the content-addressed id for this example."""

        return DATASET_VERSION_ITEM_ID_PREFIX + _content_hash({
            "input": self.input,
            "input_id": self.input_id,
            "expected_response": self.expected_response,
            "expected_contexts": self.expected_contexts,
        })

    def identity_digest(self) -> Dict[str, Any]:
        """The contribution this item makes to its version's content hash.

        Includes the per-version properties (`meta`, `splits`) so that
        republishing the same examples with different annotations produces a
        genuinely different version rather than silently returning the old one.
        """

        return {
            "item_id": self.item_id,
            "meta": self.meta,
            "splits": self.splits,
        }

    def __hash__(self):
        return hash(self.item_id)


class DatasetVersion(serial_utils.SerialModel, Hashable):
    """An immutable snapshot of a dataset's examples.

    A version is content-addressed: `dataset_version_id` is derived from the
    owning dataset, the ordered contents of the version and its source
    metadata. Publishing the same content twice therefore yields the same id
    and is idempotent. `description` and `parent_dataset_version_id` are
    provenance annotations and are deliberately excluded from the hash, so
    re-publishing identical content under a new description returns the
    existing version unchanged.
    """

    dataset_version_id: types_schema.DatasetVersionID  # str
    """The content-addressed identifier for this version."""

    dataset_id: types_schema.DatasetID  # str
    """The stable dataset this version belongs to.

    See [Dataset.dataset_id][trulens.core.schema.dataset.Dataset.dataset_id]."""

    parent_dataset_version_id: Optional[types_schema.DatasetVersionID] = None
    """The version this one was derived from, if any.

    Optional provenance only: a delta between two versions is computed from
    their contents, not from this pointer."""

    description: Optional[str] = None
    """Free-text description of the version."""

    source_meta: types_schema.Metadata = pydantic.Field(default_factory=dict)
    """Metadata describing where the version's content came from."""

    content_hash: str = ""
    """Hash of the version's ordered contents.

    `dataset_version_id` is this digest with a fixed prefix."""

    item_count: int = 0
    """Number of examples in the version."""

    created_at: float = 0.0
    """Creation timestamp, as seconds since the epoch."""

    version_index: Optional[int] = None
    """Monotonic per-dataset index, assigned when the version is persisted.

    This is the deterministic ordering key used to resolve the latest version;
    creation timestamps alone can tie. Version zero always has index 0."""

    items: List[DatasetVersionItem] = pydantic.Field(default_factory=list)
    """The examples in this version.

    Populated by loaders. Items live in their own table and are *not* part of
    the persisted version metadata; `item_count` is authoritative for a version
    whose items have not been loaded."""

    def __init__(
        self,
        dataset_id: types_schema.DatasetID,
        items: Optional[Sequence[DatasetVersionItem]] = None,
        parent_dataset_version_id: Optional[
            types_schema.DatasetVersionID
        ] = None,
        description: Optional[str] = None,
        source_meta: Optional[types_schema.Metadata] = None,
        created_at: Optional[float] = None,
        dataset_version_id: Optional[types_schema.DatasetVersionID] = None,
        content_hash: Optional[str] = None,
        item_count: Optional[int] = None,
        version_index: Optional[int] = None,
        **kwargs,
    ):
        items = list(items) if items is not None else []

        kwargs["dataset_id"] = dataset_id
        kwargs["items"] = items
        kwargs["parent_dataset_version_id"] = parent_dataset_version_id
        kwargs["description"] = description
        kwargs["source_meta"] = (
            _normalize_content(source_meta) if source_meta is not None else {}
        )
        kwargs["created_at"] = created_at if created_at is not None else 0.0
        kwargs["item_count"] = (
            item_count if item_count is not None else len(items)
        )
        kwargs["version_index"] = version_index

        super().__init__(
            dataset_version_id="temporary",
            content_hash="",
            **kwargs,
        )  # both updated below

        self.content_hash = (
            content_hash if content_hash is not None else self.compute_hash()
        )
        self.dataset_version_id = (
            dataset_version_id
            if dataset_version_id is not None
            else DATASET_VERSION_ID_PREFIX + self.content_hash
        )

        for item in self.items:
            item.dataset_version_id = self.dataset_version_id

    def compute_hash(self) -> str:
        """Compute the content hash of this version's ordered contents."""

        return _content_hash({
            "dataset_id": self.dataset_id,
            "source_meta": self.source_meta,
            "items": [item.identity_digest() for item in self.items],
        })

    def metadata_json(self, **kwargs) -> str:
        """Serialize the version metadata, excluding its items.

        Items are persisted in their own table, so storing them again inside
        the version row would duplicate the whole snapshot.
        """

        return json.dumps({
            "dataset_version_id": self.dataset_version_id,
            "dataset_id": self.dataset_id,
            "parent_dataset_version_id": self.parent_dataset_version_id,
            "description": self.description,
            "source_meta": self.source_meta,
            "content_hash": self.content_hash,
            "item_count": self.item_count,
            "created_at": self.created_at,
            "version_index": self.version_index,
        })

    @property
    def is_version_zero(self) -> bool:
        """Whether this version was reconstructed from legacy ground truth."""

        return self.source_meta.get("origin") == VERSION_ZERO_ORIGIN

    def split(self, split_name: str) -> List[DatasetVersionItem]:
        """Return the loaded items belonging to a named split."""

        return [item for item in self.items if split_name in item.splits]

    def split_names(self) -> List[str]:
        """Return every split name used by the loaded items."""

        names = {name for item in self.items for name in item.splits}
        return sorted(names)

    def __hash__(self):
        return hash(self.dataset_version_id)


class DatasetVersionDiff(serial_utils.SerialModel):
    """Membership comparison between two dataset versions."""

    dataset_version_id_a: types_schema.DatasetVersionID
    """The baseline version."""

    dataset_version_id_b: types_schema.DatasetVersionID
    """The version compared against the baseline."""

    added: List[types_schema.DatasetVersionItemID] = pydantic.Field(
        default_factory=list
    )
    """Items present in `b` but not in `a`, in their order within `b`."""

    removed: List[types_schema.DatasetVersionItemID] = pydantic.Field(
        default_factory=list
    )
    """Items present in `a` but not in `b`, in their order within `a`."""

    unchanged: List[types_schema.DatasetVersionItemID] = pydantic.Field(
        default_factory=list
    )
    """Items present in both versions, in their order within `a`."""

    @classmethod
    def between(
        cls,
        version_a: DatasetVersion,
        version_b: DatasetVersion,
    ) -> DatasetVersionDiff:
        """Compare the membership of two loaded versions."""

        ids_a = [item.item_id for item in version_a.items]
        ids_b = [item.item_id for item in version_b.items]
        set_a, set_b = set(ids_a), set(ids_b)

        return cls(
            dataset_version_id_a=version_a.dataset_version_id,
            dataset_version_id_b=version_b.dataset_version_id,
            added=[i for i in ids_b if i not in set_a],
            removed=[i for i in ids_a if i not in set_b],
            unchanged=[i for i in ids_a if i in set_b],
        )


# HACK013: Need these if using __future__.annotations .
Dataset.model_rebuild()
DatasetVersionItem.model_rebuild()
DatasetVersion.model_rebuild()
DatasetVersionDiff.model_rebuild()
