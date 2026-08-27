"""Portable semantic trace search and failure clustering.

Helpers for the `semantic_trace_analysis.ipynb` cookbook. Everything here runs
locally on standard text-analysis tools: TF-IDF over unigrams and bigrams, a
truncated SVD latent space, exact cosine similarity, and MiniBatchKMeans. There
is no vector service, no index to keep, and no network call.

The default data source is a checked-in synthetic fixture so the notebook runs
offline. The same functions accept real data loaded through public TruLens
APIs — `TruSession.get_records_and_feedback()` and `TruSession.get_events()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import logging
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

RANDOM_SEED = 20240617
"""Fixed seed so every run of the notebook produces the same output."""

FIXTURE_PATH = Path(__file__).parent / "data" / "semantic_trace_fixture.jsonl"
"""Synthetic traces used when no real session is supplied."""

RECORD_COLUMNS = [
    "record_id",
    "trace_id",
    "app_name",
    "app_version",
    "conversation_id",
    "ts",
    "input",
    "output",
    "error",
    "latency",
    "total_cost",
    "cost_currency",
    "total_tokens",
]
"""Normalized `records` columns. Metric columns are appended alongside these."""

SPAN_COLUMNS = [
    "record_id",
    "trace_id",
    "span_id",
    "parent_span_id",
    "span_type",
    "span_name",
    "duration_ms",
    "status",
]
"""Normalized `spans` columns, before selected semantic attributes."""

EVALUATION_COLUMNS = [
    "record_id",
    "metric",
    "score",
    "higher_is_better",
    "explanation",
    "status",
    "eval_cost",
]
"""Normalized `evaluations` columns."""

SPAN_ATTRIBUTE_ALLOWLIST = ("model", "tool_name")
"""Span attributes copied into `spans`.

An allowlist rather than a passthrough: span attributes are arbitrary
user-controlled data, and only these are wanted in a failure document.
"""

DOCUMENT_FIELD_ALLOWLIST = ("input", "output", "error", "evaluation", "path")
"""Sections a failure document may contain, in the order they are emitted."""

DEFAULT_TRUNCATION = 600
"""Characters kept per document section before deterministic truncation."""

TRUNCATION_MARKER = " …[truncated]"
"""Appended to any section that was shortened, so truncation is visible."""

DEFAULT_MASK_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"sk-[A-Za-z0-9\-]{8,}", "[REDACTED_API_KEY]"),
    (r"ghp_[A-Za-z0-9]{8,}", "[REDACTED_TOKEN]"),
    (r"AKIA[0-9A-Z]{12,}", "[REDACTED_AWS_KEY]"),
    (r"Bearer\s+[A-Za-z0-9._\-]{8,}", "[REDACTED_BEARER]"),
    (
        r"\b(?:password|passwd|secret|api_key|token|authorization)\s*[=:]\s*\S+",
        "[REDACTED_SECRET]",
    ),
)
"""Patterns masked before any text is vectorized.

Warning:
    This is secret redaction for well-known credential shapes, not general PII
    detection. It will not find names, addresses, account numbers, or anything
    else that does not match these patterns. Review your own data before
    sharing vectors or cluster summaries derived from it.
"""


# --------------------------------------------------------------------------
# Loading and normalization
# --------------------------------------------------------------------------


def load_fixture(path: str | Path = FIXTURE_PATH) -> dict[str, pd.DataFrame]:
    """Load the synthetic trace fixture.

    Args:
        path: JSONL file whose rows each carry a `kind` of `record`, `span` or
            `evaluation`.

    Returns:
        `{"records": ..., "spans": ..., "evaluations": ...}`, normalized.
    """

    buckets: dict[str, list[dict[str, Any]]] = {
        "record": [],
        "span": [],
        "evaluation": [],
    }

    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            kind = row.pop("kind", None)
            if kind not in buckets:
                raise ValueError(
                    f"Unknown fixture row kind {kind!r} in {path}. Expected one "
                    f"of {', '.join(sorted(buckets))}."
                )
            buckets[kind].append(row)

    return {
        "records": normalize_records(pd.DataFrame(buckets["record"])),
        "spans": normalize_spans(pd.DataFrame(buckets["span"])),
        "evaluations": normalize_evaluations(
            pd.DataFrame(buckets["evaluation"])
        ),
    }


def _with_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return `frame` with `columns` first, adding any that are absent."""

    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = None
    rest = [c for c in out.columns if c not in columns]
    return out[list(columns) + rest]


def normalize_records(records: pd.DataFrame) -> pd.DataFrame:
    """Normalize a records frame into the shape the helpers expect.

    Accepts either the fixture shape or what
    `TruSession.get_records_and_feedback()` returns; metric columns present in
    the input are preserved alongside the standard columns.
    """

    if records.empty:
        return _with_columns(pd.DataFrame(), RECORD_COLUMNS)

    out = _with_columns(records, RECORD_COLUMNS)
    out["input"] = out["input"].map(_as_text)
    out["output"] = out["output"].map(_as_text)
    out["error"] = out["error"].map(lambda v: _as_text(v) or None)
    return out.reset_index(drop=True)


def normalize_spans(spans: pd.DataFrame) -> pd.DataFrame:
    """Normalize a spans frame, keeping only allowlisted attributes."""

    if spans.empty:
        return _with_columns(pd.DataFrame(), SPAN_COLUMNS)

    out = _with_columns(spans, SPAN_COLUMNS)

    attributes = out["attributes"] if "attributes" in out.columns else None
    for name in SPAN_ATTRIBUTE_ALLOWLIST:
        if name in out.columns:
            continue
        if attributes is None:
            out[name] = None
        else:
            out[name] = [
                (a or {}).get(name) if isinstance(a, Mapping) else None
                for a in attributes
            ]

    if "attributes" in out.columns:
        out = out.drop(columns=["attributes"])

    return out.reset_index(drop=True)


def normalize_evaluations(evaluations: pd.DataFrame) -> pd.DataFrame:
    """Normalize an evaluations frame."""

    if evaluations.empty:
        return _with_columns(pd.DataFrame(), EVALUATION_COLUMNS)

    out = _with_columns(evaluations, EVALUATION_COLUMNS)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out["higher_is_better"] = out["higher_is_better"].map(
        lambda v: True if v is None else bool(v)
    )
    return out.reset_index(drop=True)


def evaluations_of_records(
    records: pd.DataFrame,
    metric_columns: Sequence[str],
    directions: Mapping[str, bool] | None = None,
) -> pd.DataFrame:
    """Derive an evaluations frame from metric columns on a records frame.

    `get_records_and_feedback()` returns one column per metric plus a
    `"<metric> direction"` column, rather than long-form evaluations; this
    reshapes that into the `evaluations` frame the helpers use.
    """

    directions = dict(directions or {})
    rows = []

    for _, record in records.iterrows():
        for metric in metric_columns:
            if metric not in records.columns:
                continue
            score = record[metric]
            if _is_missing(score):
                continue

            direction_column = f"{metric} direction"
            if metric in directions:
                higher_is_better = directions[metric]
            elif direction_column in records.index:
                higher_is_better = bool(record[direction_column])
            else:
                higher_is_better = True

            rows.append({
                "record_id": record["record_id"],
                "metric": metric,
                "score": score,
                "higher_is_better": higher_is_better,
                "explanation": record.get(f"{metric} explanation"),
                "status": "DONE",
                "eval_cost": record.get(f"{metric} feedback cost in USD"),
            })

    return normalize_evaluations(pd.DataFrame(rows))


def load_from_session(
    session: Any,
    app_name: str | None = None,
    limit: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Load real traces through public TruLens APIs.

    Uses only `get_records_and_feedback()` and, when available, `get_events()`.
    No underscore-prefixed database or connector method is touched.

    Args:
        session: A `TruSession`.
        app_name: Restrict to one app.
        limit: Maximum records to pull.

    Returns:
        The same three normalized frames `load_fixture` returns.
    """

    kwargs: dict[str, Any] = {}
    if app_name is not None:
        kwargs["app_name"] = app_name
    if limit is not None:
        kwargs["limit"] = limit

    records, metric_columns = session.get_records_and_feedback(**kwargs)
    records = normalize_records(records)

    spans = pd.DataFrame()
    try:
        events = session.get_events(**kwargs)
    except Exception as e:  # pragma: no cover - depends on backend support
        logger.info(
            "Raw spans are unavailable for this session (%s: %s); continuing "
            "with records and evaluations only.",
            type(e).__name__,
            e,
        )
    else:
        spans = _spans_of_events(events)

    return {
        "records": records,
        "spans": normalize_spans(spans),
        "evaluations": evaluations_of_records(records, list(metric_columns)),
    }


def _spans_of_events(events: pd.DataFrame) -> pd.DataFrame:
    """Reshape an OTEL events frame into the `spans` shape."""

    if events is None or len(events) == 0:
        return pd.DataFrame()

    rows = []
    for _, event in events.iterrows():
        attributes = event.get("record_attributes") or {}
        if not isinstance(attributes, Mapping):
            attributes = {}
        record = event.get("record") or {}
        if not isinstance(record, Mapping):
            record = {}

        rows.append({
            "record_id": attributes.get("ai.observability.record_id"),
            "trace_id": (event.get("trace") or {}).get("trace_id")
            if isinstance(event.get("trace"), Mapping)
            else None,
            "span_id": record.get("span_id")
            or (event.get("trace") or {}).get("span_id")
            if isinstance(event.get("trace"), Mapping)
            else None,
            "parent_span_id": (event.get("trace") or {}).get("parent_id")
            if isinstance(event.get("trace"), Mapping)
            else None,
            "span_type": attributes.get("ai.observability.span_type"),
            "span_name": record.get("name"),
            "duration_ms": _duration_ms(event),
            "status": record.get("status"),
            "attributes": {
                "model": attributes.get("ai.observability.cost.model"),
                "tool_name": attributes.get("ai.observability.call.function"),
            },
        })

    return pd.DataFrame(rows)


def _duration_ms(event: Mapping[str, Any]) -> float | None:
    """Span duration in milliseconds, when both timestamps are present."""

    start, end = event.get("start_timestamp"), event.get("timestamp")
    if start is None or end is None:
        return None
    try:
        return (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() * 1000
    except (TypeError, ValueError):
        return None


def _is_missing(value: Any) -> bool:
    """Whether a dataframe value should be treated as absent.

    Pandas represents an absent object value as NaN, and `bool(float("nan"))`
    is `True`, so truthiness alone silently treats every empty cell as present.
    """

    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _as_text(value: Any) -> str:
    """Render a recorded value as stable text.

    Inputs and outputs arrive as strings in OTEL mode and as parsed JSON in the
    legacy schema; dicts and lists are serialized with sorted keys so the same
    content always produces the same document.
    """

    if _is_missing(value):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value).strip()


# --------------------------------------------------------------------------
# Failure selection and documents
# --------------------------------------------------------------------------


def select_failures(
    records: pd.DataFrame,
    evaluations: pd.DataFrame,
    metric: str | None = None,
    threshold: float = 0.5,
    include_errors: bool = True,
) -> pd.DataFrame:
    """Select the records worth analyzing.

    A record qualifies when it errored or when `metric` scores it badly.
    "Badly" respects the metric's direction: below `threshold` for a
    higher-is-better metric, above it for a lower-is-better one.

    Args:
        records: Normalized records.
        evaluations: Normalized evaluations.
        metric: Metric to judge on. Every metric is considered when omitted.
        threshold: Score boundary, in the metric's own units.
        include_errors: Whether errored records qualify regardless of score.

    Returns:
        The qualifying records, with `failure_reason` explaining each one.
    """

    if records.empty:
        return records.assign(failure_reason=[])

    scored = evaluations
    if metric is not None:
        scored = scored[scored["metric"] == metric]

    reasons: dict[str, list[str]] = {}

    for _, evaluation in scored.iterrows():
        score = evaluation["score"]
        if _is_missing(score):
            continue

        higher_is_better = bool(evaluation["higher_is_better"])
        failed = score < threshold if higher_is_better else score > threshold
        if not failed:
            continue

        comparison = "<" if higher_is_better else ">"
        reasons.setdefault(evaluation["record_id"], []).append(
            f"{evaluation['metric']} {comparison} {threshold}"
        )

    if include_errors:
        for _, record in records.iterrows():
            if not _is_missing(record["error"]):
                reasons.setdefault(record["record_id"], []).insert(
                    0, "record errored"
                )

    selected = records[records["record_id"].isin(reasons)].copy()
    selected["failure_reason"] = [
        "; ".join(sorted(reasons[record_id]))
        for record_id in selected["record_id"]
    ]

    return selected.reset_index(drop=True)


def mask_text(
    text: str,
    patterns: Sequence[tuple[str, str]] = DEFAULT_MASK_PATTERNS,
) -> str:
    """Replace known credential shapes before the text is vectorized.

    Warning:
        Secret redaction, not PII detection. See
        [DEFAULT_MASK_PATTERNS][].
    """

    masked = text
    for pattern, replacement in patterns:
        masked = re.sub(pattern, replacement, masked, flags=re.IGNORECASE)
    return masked


def truncate(text: str, limit: int = DEFAULT_TRUNCATION) -> str:
    """Shorten `text` to `limit` characters, marking that it was shortened.

    Deterministic by design: the same input always yields the same output, so
    documents stay stable across runs.
    """

    if limit < 0:
        raise ValueError(f"`limit` must not be negative, got {limit}.")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + TRUNCATION_MARKER


def span_path(spans: pd.DataFrame, record_id: str, limit: int = 12) -> str:
    """The span types a record passed through, in order, deduplicated.

    Repeated spans of the same type collapse to one entry so that a retry loop
    does not dominate a document.
    """

    if spans.empty:
        return ""

    rows = spans[spans["record_id"] == record_id]
    if rows.empty:
        return ""

    ordered = []
    for span_type in rows["span_type"]:
        if span_type and (not ordered or ordered[-1] != span_type):
            ordered.append(str(span_type))

    return " > ".join(ordered[:limit])


@dataclass
class DocumentConfig:
    """How failure documents are built."""

    fields: Sequence[str] = DOCUMENT_FIELD_ALLOWLIST
    """Sections to include, from `DOCUMENT_FIELD_ALLOWLIST`."""

    truncation: int = DEFAULT_TRUNCATION
    """Characters kept per section."""

    mask_patterns: Sequence[tuple[str, str]] = DEFAULT_MASK_PATTERNS
    """Applied to every section before it enters the document."""

    max_evaluations: int = 4
    """Evaluation lines kept per document, worst first."""

    def __post_init__(self):
        unknown = [f for f in self.fields if f not in DOCUMENT_FIELD_ALLOWLIST]
        if unknown:
            raise ValueError(
                f"Unknown document field(s): {', '.join(unknown)}. Allowed: "
                f"{', '.join(DOCUMENT_FIELD_ALLOWLIST)}."
            )


def build_failure_document(
    record: Mapping[str, Any],
    evaluations: pd.DataFrame,
    spans: pd.DataFrame,
    config: DocumentConfig | None = None,
) -> str:
    """Build one deterministic document describing a failed record.

    Identifiers, scores, costs, timestamps, versions, models and tools stay out
    of the text and travel as metadata instead: embedding them would let a
    record id or a cost figure influence similarity, which is not what "similar
    failure" should mean.
    """

    config = config or DocumentConfig()
    record_id = record["record_id"]
    sections: list[str] = []

    def _clean(value: Any) -> str:
        return truncate(
            mask_text(_as_text(value), config.mask_patterns), config.truncation
        )

    for field_name in config.fields:
        if field_name == "input":
            sections.append(f"[INPUT] {_clean(record.get('input'))}")

        elif field_name == "output":
            sections.append(f"[OUTPUT] {_clean(record.get('output'))}")

        elif field_name == "error":
            error = record.get("error")
            if not _is_missing(error):
                sections.append(f"[ERROR] {_clean(error)}")

        elif field_name == "evaluation":
            sections.extend(
                _evaluation_lines(record_id, evaluations, config, _clean)
            )

        elif field_name == "path":
            path = span_path(spans, record_id)
            if path:
                sections.append(f"[PATH] {path}")

    return "\n".join(sections)


def _evaluation_lines(
    record_id: str,
    evaluations: pd.DataFrame,
    config: DocumentConfig,
    clean: Callable[[Any], str],
) -> list[str]:
    """The `[EVALUATION]` lines for one record, worst score first."""

    if evaluations.empty:
        return []

    rows = evaluations[evaluations["record_id"] == record_id]
    if rows.empty:
        return []

    # Worst first, so truncation drops the least interesting evaluations, and
    # by metric name so equal scores never reorder between runs.
    ordered = rows.assign(
        _severity=[
            -score if higher_is_better else score
            for score, higher_is_better in zip(
                rows["score"].fillna(0.0), rows["higher_is_better"]
            )
        ]
    ).sort_values(["_severity", "metric"], ascending=[False, True])

    lines = []
    for _, evaluation in ordered.head(config.max_evaluations).iterrows():
        explanation = clean(evaluation["explanation"])
        lines.append(
            f"[EVALUATION] metric={evaluation['metric']}; "
            f"explanation={explanation}"
        )
    return lines


def build_failure_documents(
    failures: pd.DataFrame,
    evaluations: pd.DataFrame,
    spans: pd.DataFrame,
    config: DocumentConfig | None = None,
) -> pd.DataFrame:
    """Build a document per failing record, with its metadata alongside.

    Returns:
        A frame of `record_id`, `document`, and the metadata columns that were
        deliberately kept out of the document text.
    """

    config = config or DocumentConfig()
    metadata_columns = [
        c
        for c in (
            "record_id",
            "app_name",
            "app_version",
            "conversation_id",
            "ts",
            "latency",
            "total_cost",
            "cost_currency",
            "total_tokens",
            "model",
            "tool",
            "error",
            "failure_reason",
            "failure_group",
        )
        if c in failures.columns
    ]

    rows = []
    for _, record in failures.iterrows():
        rows.append({
            **{c: record[c] for c in metadata_columns},
            "document": build_failure_document(
                record, evaluations, spans, config
            ),
        })

    documents = pd.DataFrame(rows)
    if documents.empty:
        return documents

    # Stable order regardless of how the failures frame arrived.
    return documents.sort_values("record_id").reset_index(drop=True)


# --------------------------------------------------------------------------
# Semantic search
# --------------------------------------------------------------------------


@dataclass
class SearchResult:
    """One retrieved failure."""

    record_id: str
    score: float
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticIndex:
    """TF-IDF plus truncated SVD, searched by exact cosine similarity.

    Small enough that exact search is the right answer: an approximate
    nearest-neighbor structure would add a dependency and a failure mode
    without changing the result at cookbook scale.
    """

    def __init__(
        self,
        n_components: int = 64,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        random_state: int = RANDOM_SEED,
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        self.svd: TruncatedSVD | None = None
        self.vectors: np.ndarray | None = None
        self.documents: pd.DataFrame | None = None

    def fit(self, documents: pd.DataFrame) -> "SemanticIndex":
        """Fit on a documents frame from `build_failure_documents`."""

        if documents.empty:
            raise ValueError("Cannot build an index from zero documents.")

        self.documents = documents.reset_index(drop=True)
        counts = self.vectorizer.fit_transform(self.documents["document"])

        # SVD needs strictly fewer components than features.
        components = min(self.n_components, max(counts.shape[1] - 1, 1))
        self.svd = TruncatedSVD(
            n_components=components, random_state=self.random_state
        )
        self.vectors = normalize(self.svd.fit_transform(counts))
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Project raw text into the fitted latent space."""

        if self.svd is None:
            raise RuntimeError("The index has not been fitted yet.")
        masked = [mask_text(_as_text(t)) for t in texts]
        return normalize(self.svd.transform(self.vectorizer.transform(masked)))

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return the `k` failures most similar in meaning to `query`."""

        if self.vectors is None or self.documents is None:
            raise RuntimeError("The index has not been fitted yet.")
        if k < 1:
            raise ValueError(f"`k` must be positive, got {k}.")

        # Vectors are L2-normalized, so the dot product is the cosine.
        similarities = self.vectors @ self.transform([query])[0]
        # Ties break by position, keeping results stable across runs.
        order = np.argsort(-similarities, kind="stable")[:k]

        results = []
        for position in order:
            row = self.documents.iloc[position]
            results.append(
                SearchResult(
                    record_id=row["record_id"],
                    score=float(similarities[position]),
                    document=row["document"],
                    metadata={
                        c: row[c]
                        for c in self.documents.columns
                        if c != "document"
                    },
                )
            )
        return results


def evaluate_search(
    index: SemanticIndex,
    labeled_queries: Sequence[Mapping[str, Any]],
    k: int = 5,
    group_column: str = "failure_group",
) -> pd.DataFrame:
    """Score an index against labeled queries.

    Reports success@k and recall@k per query rather than a hand-picked example,
    so a change that makes search worse is visible.

    Args:
        index: A fitted index.
        labeled_queries: Each `{"query": ..., "failure_group": ...}`.
        k: How many results to consider.
        group_column: Metadata column holding the failure group.

    Returns:
        One row per query, plus the hit count and recall.
    """

    if index.documents is None:
        raise RuntimeError("The index has not been fitted yet.")

    rows = []
    for labeled in labeled_queries:
        query, expected = labeled["query"], labeled[group_column]
        results = index.search(query, k=k)
        groups = [r.metadata.get(group_column) for r in results]
        hits = sum(1 for g in groups if g == expected)
        relevant = int((index.documents[group_column] == expected).sum())

        rows.append({
            "query": query,
            "expected_group": expected,
            "hits_at_k": hits,
            "success_at_k": hits > 0,
            "recall_at_k": hits / min(k, relevant)
            if relevant
            else float("nan"),
            "top_group": groups[0] if groups else None,
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Failure clustering
# --------------------------------------------------------------------------


def cluster_vectors(
    vectors: np.ndarray, k: int, random_state: int = RANDOM_SEED
) -> np.ndarray:
    """Assign each vector to one of `k` clusters, deterministically."""

    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,
        batch_size=min(256, max(len(vectors), 1)),
    )
    return model.fit_predict(vectors)


def stability(
    vectors: np.ndarray, k: int, seeds: Sequence[int] = (0, 1, 2, 3)
) -> float:
    """Mean adjusted Rand index across repeated seeds.

    A clustering that reshuffles when only the seed changes is not describing
    structure in the data, so this is reported next to silhouette rather than
    trusting silhouette alone.
    """

    labelings = [
        cluster_vectors(vectors, k, random_state=RANDOM_SEED + seed)
        for seed in seeds
    ]

    scores = [
        adjusted_rand_score(labelings[i], labelings[j])
        for i in range(len(labelings))
        for j in range(i + 1, len(labelings))
    ]
    return float(np.mean(scores)) if scores else 1.0


def select_k(
    vectors: np.ndarray,
    candidates: Iterable[int] = (2, 3, 4, 5, 6),
    seeds: Sequence[int] = (0, 1, 2, 3),
) -> pd.DataFrame:
    """Score each candidate `k` on silhouette, stability and size spread.

    Returns:
        One row per candidate, best first by silhouette then stability. Nothing
        here picks `k` for you: the point is to see the trade-off.
    """

    rows = []
    for k in candidates:
        if k < 2 or k >= len(vectors):
            continue

        labels = cluster_vectors(vectors, k)
        sizes = np.bincount(labels, minlength=k)

        rows.append({
            "k": k,
            "silhouette": float(silhouette_score(vectors, labels)),
            "stability": stability(vectors, k, seeds=seeds),
            "smallest_cluster": int(sizes.min()),
            "largest_cluster": int(sizes.max()),
            "empty_clusters": int((sizes == 0).sum()),
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["silhouette", "stability"], ascending=False
    ).reset_index(drop=True)


def medoid_index(vectors: np.ndarray, members: Sequence[int]) -> int:
    """Index of the member closest to its cluster's centre.

    A real record rather than a synthetic centroid, so the summary always
    points at something a reader can open.
    """

    if len(members) == 0:
        raise ValueError("A cluster must have at least one member.")

    member_vectors = vectors[list(members)]
    centre = member_vectors.mean(axis=0)
    distances = np.linalg.norm(member_vectors - centre, axis=1)
    # Ties resolve to the earliest member, keeping the choice deterministic.
    return int(members[int(np.argmin(distances))])


def top_terms(
    index: SemanticIndex, members: Sequence[int], n: int = 8
) -> list[str]:
    """The terms that most characterize a cluster."""

    if index.documents is None:
        raise RuntimeError("The index has not been fitted yet.")

    counts = index.vectorizer.transform(
        index.documents.iloc[list(members)]["document"]
    )
    weights = np.asarray(counts.mean(axis=0)).ravel()
    names = index.vectorizer.get_feature_names_out()

    order = np.argsort(-weights, kind="stable")[:n]
    return [str(names[i]) for i in order if weights[i] > 0]


def summarize_clusters(
    index: SemanticIndex,
    labels: np.ndarray,
    evaluations: pd.DataFrame,
    n_terms: int = 8,
    n_examples: int = 2,
) -> pd.DataFrame:
    """Describe each cluster in terms a reader can act on.

    Warning:
        Clusters are unsupervised. They are a lead to investigate, not ground
        truth about failure modes.
    """

    if index.documents is None or index.vectors is None:
        raise RuntimeError("The index has not been fitted yet.")

    documents = index.documents
    rows = []

    for cluster in sorted(set(int(c) for c in labels)):
        members = [i for i, c in enumerate(labels) if int(c) == cluster]
        medoid = medoid_index(index.vectors, members)
        member_rows = documents.iloc[members]
        member_ids = list(member_rows["record_id"])

        scores = evaluations[evaluations["record_id"].isin(member_ids)]
        worst = (
            scores
            .assign(
                _severity=[
                    -s if h else s
                    for s, h in zip(
                        scores["score"].fillna(0.0), scores["higher_is_better"]
                    )
                ]
            )
            .sort_values("_severity", ascending=False)
            .head(3)
        )

        rows.append({
            "cluster": cluster,
            "size": len(members),
            "medoid_record_id": documents.iloc[medoid]["record_id"],
            "top_terms": top_terms(index, members, n=n_terms),
            "app_versions": _unique(member_rows, "app_version"),
            "models": _unique(member_rows, "model"),
            "tools": _unique(member_rows, "tool"),
            "errors": _unique(member_rows, "error"),
            "lowest_metrics": [
                f"{m}={s:.2f}" for m, s in zip(worst["metric"], worst["score"])
            ],
            "examples": member_ids[:n_examples],
        })

    return pd.DataFrame(rows)


def _unique(frame: pd.DataFrame, column: str) -> list[str]:
    """Sorted distinct non-empty values of a column."""

    if column not in frame.columns:
        return []
    values = {str(v) for v in frame[column] if not _is_missing(v)}
    return sorted(values)


def pca_2d(vectors: np.ndarray, random_state: int = RANDOM_SEED) -> np.ndarray:
    """Two-dimensional projection, for plotting only.

    Never used as clustering input: clusters come from the full latent space,
    and this is just how they get drawn.
    """

    if len(vectors) < 2:
        return np.zeros((len(vectors), 2))
    return PCA(n_components=2, random_state=random_state).fit_transform(vectors)
