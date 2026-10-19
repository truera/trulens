"""SQL result-set comparator.

Compares two SQL result sets (as pandas DataFrames) using:
1. Exact equality check
2. LLM-based column name alignment (Cortex Complete)
3. Multi-strategy row normalization (null removal, sorting)
4. Column comparison (exact, approximate/epsilon, categorical bijection)

Returns a ComparisonResult with Equal/NotEqual/Unclear verdict and diagnostic messages.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dateutil import parser as dateutil_parser

from src.services.config import SNOWFLAKE_ACCOUNT_URL, SNOWFLAKE_PAT

logger = logging.getLogger(__name__)

FLOAT_EPSILON = 0.01
FLOAT_REL_TOL = 1e-05
MAX_CATEGORICAL = 250


class Equality(str, Enum):
    EQUAL = "Equal"
    NOT_EQUAL = "Not equal"
    UNCLEAR = "Need manual analysis"
    NOT_RUN = "Comparison was not run"


@dataclass
class ComparisonResult:
    result: Equality = Equality.NOT_RUN
    info: list[str] = field(default_factory=list)
    warn: list[str] = field(default_factory=list)
    err: list[str] = field(default_factory=list)


COLUMN_MAPPING_PROMPT = """You are given two arrays representing column names from different dataframes. These columns are intended to represent the same underlying data, but the names may vary due to differences in naming conventions, abbreviations, or use of SQL functions. Your task is to align them by generating a JSON object that maps each column from the first array (cols1) to the most semantically similar column in the second array (cols2), following these rules:

 - Each column in cols1 must map to at most one column in cols2.
 - If no suitable match exists in cols2, map the column to null — but only if you're confident there is no match.
 - Match based on semantic similarity, not just exact or partial string matches.
 - Do not reuse columns from cols2.
 - Do not include any explanations. Only output the JSON object.

Example:
cols1 = ['age', 'name', 'region_name', 'nation_key']
cols2 = ['first_name', 'region_key', 'availability', 'nation']

Output:
{
  "age": null,
  "name": "first_name",
  "region_name": "region_key",
  "nation_key": "nation"
}

Now, align the following:
cols1 = [<COLS_1>]
cols2 = [<COLS_2>]
"""


def _format_cols(cols: list[str]) -> str:
    return ", ".join(f"'{c}'" for c in cols)


def _build_column_mapping_prompt(cols1: list[str], cols2: list[str]) -> str:
    prompt = COLUMN_MAPPING_PROMPT.replace("<COLS_1>", _format_cols(cols1))
    prompt = prompt.replace("<COLS_2>", _format_cols(cols2))
    return prompt


def _llm_column_mapping(
    cols1: list[str],
    cols2: list[str],
    model: str = "llama3.1-70b",
) -> dict[str, Optional[str]]:
    prompt = _build_column_mapping_prompt(cols1, cols2)
    resp = requests.post(
        f"{SNOWFLAKE_ACCOUNT_URL}/api/v2/cortex/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {SNOWFLAKE_PAT}",
            "Content-Type": "application/json",
            "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        },
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"LLM response is not valid JSON: {text}")
    raw = json.loads(text[start:end])
    result: dict[str, Optional[str]] = {}
    for k, v in raw.items():
        if v is None:
            result[k] = None
        elif isinstance(v, str):
            result[k] = v
        else:
            raise ValueError(f"Invalid value type for key {k!r}: {type(v)}")
    return result


def _is_null(val) -> bool:
    if pd.isna(val):
        return True
    if isinstance(val, str) and val.strip().lower() in ("", "null"):
        return True
    return False


def _is_date_column(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    for val in series.dropna():
        try:
            dateutil_parser.parse(str(val))
        except (ValueError, TypeError):
            return False
    return True


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        non_null_orig = df[col].dropna()
        non_null_orig = non_null_orig[~non_null_orig.isin(["", "null", "NULL"])]
        non_null_converted = converted.loc[non_null_orig.index]
        if non_null_orig.empty or non_null_converted.notna().all():
            if non_null_converted.dropna().apply(lambda x: x == math.trunc(x) if not math.isnan(x) else True).all():
                df[col] = converted.astype("Int64")
            else:
                df[col] = converted
    return df


@dataclass
class _NormalizeColumnsResult:
    gold: pd.DataFrame
    pred: pd.DataFrame
    columns_mapping: dict[str, tuple[str, str]] = field(default_factory=dict)
    info: list[str] = field(default_factory=list)
    warn: list[str] = field(default_factory=list)
    err: list[str] = field(default_factory=list)


def _normalize_columns(
    gold: pd.DataFrame,
    pred: pd.DataFrame,
    model: str,
) -> _NormalizeColumnsResult:
    result = _NormalizeColumnsResult(gold=gold, pred=pred)
    cols1 = list(gold.columns)
    cols2 = list(pred.columns)

    try:
        mapping = _llm_column_mapping(cols1, cols2, model=model)
    except Exception as e:
        result.err.append(f"Column mapping failed: {e}")
        return result

    for col1 in cols1:
        if col1 in mapping and mapping[col1] is None:
            result.warn.append(f"Column '{col1}' from expected dataset not found in output")
            del mapping[col1]

    cols2_used = set(v for v in mapping.values() if v is not None)
    unused = [c for c in cols2 if c not in cols2_used]
    if unused:
        result.info.append(f"Unused columns in output: {unused}")

    aligned1, aligned2 = [], []
    for col1 in cols1:
        if col1 in mapping and mapping[col1] is not None:
            aligned1.append(col1)
            aligned2.append(mapping[col1])

    if not aligned1:
        result.err.append("No columns to compare after alignment")
        return result

    new_gold = gold[aligned1].copy()
    new_pred = pred[aligned2].copy()
    for i, (c1, c2) in enumerate(zip(aligned1, aligned2)):
        col_name = f"col_{i}"
        result.columns_mapping[col_name] = (c1, c2)

    new_gold.columns = [f"col_{i}" for i in range(len(aligned1))]
    new_pred.columns = [f"col_{i}" for i in range(len(aligned2))]
    result.gold = new_gold
    result.pred = new_pred
    return result


def _is_close(a: float, b: float, atol: float = FLOAT_EPSILON) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0):
        return True
    if math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) <= atol + FLOAT_REL_TOL * abs(b)


def _count_unique_excluding_null(series: pd.Series) -> int:
    return series.dropna().nunique()


class SqlResultComparator:
    def __init__(
        self,
        ignore_null: bool = False,
        ignore_duplicates: bool = False,
        attempt_sorting: bool = False,
        model: str = "llama3.1-70b",
        col_norm: Optional[_NormalizeColumnsResult] = None,
    ):
        self.ignore_null = ignore_null
        self.ignore_duplicates = ignore_duplicates
        self.attempt_sorting = attempt_sorting
        self.model = model
        self.float_epsilon = FLOAT_EPSILON
        self.max_categorical = MAX_CATEGORICAL
        self.info: list[str] = []
        self.warn: list[str] = []
        self.err: list[str] = []
        self.columns_mapping: dict[str, tuple[str, str]] = {}
        if col_norm:
            self.columns_mapping = col_norm.columns_mapping
            self.info.extend(col_norm.info)
            self.warn.extend(col_norm.warn)
            self.err.extend(col_norm.err)

    def _is_potentially_categorical(
        self, col1: pd.Series, col2: pd.Series
    ) -> bool:
        u1 = col1.nunique(dropna=False)
        u2 = col2.nunique(dropna=False)
        if u1 != u2:
            return False
        if u1 > self.max_categorical:
            return False
        if _is_date_column(col1) or _is_date_column(col2):
            return False
        if pd.api.types.is_numeric_dtype(col1) and pd.api.types.is_numeric_dtype(col2):
            return False
        if pd.api.types.is_float_dtype(col1) or pd.api.types.is_float_dtype(col2):
            return False
        return True

    def _select_sorting_order(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> tuple[list[str], list[str]]:
        non_cat = []
        cat = []
        for col in df1.columns:
            s1, s2 = df1[col], df2[col]
            if not self._is_potentially_categorical(s1, s2):
                cnt = _count_unique_excluding_null(s1) + _count_unique_excluding_null(s2)
                non_cat.append((col, cnt))
            else:
                cat.append(col)
        non_cat.sort(key=lambda x: (-x[1], x[0]))
        return [c for c, _ in non_cat], cat

    def _normalize_rows(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        d1, d2 = df1.copy(), df2.copy()
        l1, l2 = len(d1), len(d2)

        if self.ignore_null:
            mask1 = d1.apply(lambda row: not any(_is_null(v) for v in row), axis=1)
            mask2 = d2.apply(lambda row: not any(_is_null(v) for v in row), axis=1)
            filtered1 = d1[mask1]
            filtered2 = d2[mask2]
            if len(filtered1) != l1:
                self.info.append(f"Filtered {l1 - len(filtered1)} null rows from expected")
            if len(filtered2) != l2:
                self.info.append(f"Filtered {l2 - len(filtered2)} null rows from output")
            d1, d2 = filtered1, filtered2
            l1, l2 = len(d1), len(d2)

        if self.attempt_sorting:
            non_cat, cat = self._select_sorting_order(d1, d2)
            if non_cat:
                sort_cols = non_cat + cat
                d1 = d1.sort_values(sort_cols, ignore_index=True)
                d2 = d2.sort_values(sort_cols, ignore_index=True)

        if self.ignore_duplicates:
            before1 = len(d1)
            d1 = d1.drop_duplicates(ignore_index=True)
            before2 = len(d2)
            d2 = d2.drop_duplicates(ignore_index=True)
            if len(d1) != before1:
                self.info.append(f"Filtered {before1 - len(d1)} duplicate rows from expected")
            if len(d2) != before2:
                self.info.append(f"Filtered {before2 - len(d2)} duplicate rows from output")
            l1, l2 = len(d1), len(d2)

        if l1 != l2:
            self.err.append(f"Number of rows are different: expected {l1}, got {l2}")
        if l1 == 0:
            self.err.append("Expected dataset is empty")

        return d1.reset_index(drop=True), d2.reset_index(drop=True)

    def _are_columns_same(self, col1: pd.Series, col2: pd.Series) -> bool:
        if len(col1) != len(col2):
            return False
        return col1.equals(col2)

    def _are_columns_approx_same(self, col1: pd.Series, col2: pd.Series) -> bool:
        if len(col1) != len(col2):
            return False
        try:
            f1 = pd.to_numeric(col1, errors="raise").astype(float)
            f2 = pd.to_numeric(col2, errors="raise").astype(float)
        except (ValueError, TypeError):
            return self._are_columns_same(col1, col2)
        return all(
            _is_close(a, b, self.float_epsilon)
            for a, b in zip(f1, f2)
        )

    def _are_categories_match(self, col1: pd.Series, col2: pd.Series) -> bool:
        if len(col1) != len(col2):
            return False
        if not self._is_potentially_categorical(col1, col2):
            return False
        fwd: dict = {}
        rev: dict = {}
        for v1, v2 in zip(col1, col2):
            s1, s2 = str(v1), str(v2)
            if s1 in fwd:
                if fwd[s1] != s2:
                    return False
            else:
                fwd[s1] = s2
            if s2 in rev:
                if rev[s2] != s1:
                    return False
            else:
                rev[s2] = s1
        return True

    def _compare_each_row(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> bool:
        col_names = list(df1.columns)
        remaining = set(col_names)

        def trim(compare_fn, msg_fn):
            for col in col_names:
                if col not in remaining:
                    continue
                s1, s2 = df1[col], df2[col]
                if compare_fn(s1, s2):
                    orig = self.columns_mapping.get(col, (col, col))
                    self.info.append(msg_fn(orig[0], orig[1]))
                    remaining.discard(col)

        trim(
            self._are_columns_same,
            lambda c1, c2: f"Column '{c1}' matched '{c2}' exactly",
        )
        if not remaining:
            return True

        trim(
            self._are_columns_approx_same,
            lambda c1, c2: f"Column '{c1}' matched '{c2}' approximately",
        )
        if not remaining:
            return True

        trim(
            self._are_categories_match,
            lambda c1, c2: f"Column '{c1}' matched '{c2}' categorically",
        )
        if not remaining:
            return True

        for col in col_names:
            if col in remaining:
                orig = self.columns_mapping.get(col, (col, col))
                self.err.append(f"Column '{orig[0]}' differs from '{orig[1]}'")
        return False

    def compare(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> ComparisonResult:
        def _result(r: Equality) -> ComparisonResult:
            return ComparisonResult(result=r, info=self.info, warn=self.warn, err=self.err)

        df1, df2 = self._normalize_rows(df1, df2)
        if self.err:
            return _result(Equality.NOT_EQUAL)

        ok = self._compare_each_row(df1, df2)

        if self.err or not ok:
            res = Equality.NOT_EQUAL
        elif self.warn:
            res = Equality.UNCLEAR
        else:
            res = Equality.EQUAL
        return _result(res)


@dataclass
class _ResultScore:
    equality_rank: int
    error_count: int
    warn_count: int

    def __lt__(self, other: _ResultScore) -> bool:
        if self.equality_rank != other.equality_rank:
            return self.equality_rank < other.equality_rank
        if self.error_count != other.error_count:
            return self.error_count < other.error_count
        return self.warn_count < other.warn_count


def _res_to_score(r: ComparisonResult) -> _ResultScore:
    rank_map = {
        Equality.EQUAL: 0,
        Equality.UNCLEAR: 1,
        Equality.NOT_EQUAL: 2,
        Equality.NOT_RUN: 3,
    }
    rank = rank_map[r.result]
    penalty = 0
    if r.result == Equality.NOT_EQUAL:
        for msg in r.err:
            if "Number of rows are different" in msg:
                penalty = 100
                break
    return _ResultScore(
        equality_rank=rank,
        error_count=len(r.err) + penalty,
        warn_count=len(r.warn),
    )


def compare_dataframes(
    gold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    model: str = "llama3.1-70b",
) -> ComparisonResult:
    gold = convert_to_numeric(gold_df)
    pred = convert_to_numeric(pred_df)

    if gold.equals(pred):
        info = [f"Column '{c}' matched exactly" for c in gold.columns]
        return ComparisonResult(result=Equality.EQUAL, info=info)

    col_norm = _normalize_columns(gold, pred, model=model)
    if col_norm.err:
        return ComparisonResult(
            result=Equality.NOT_EQUAL,
            info=col_norm.info,
            warn=col_norm.warn,
            err=col_norm.err,
        )

    configs = [
        {"ignore_null": False, "attempt_sorting": False, "explanation": "Dataframes are not modified"},
        {"ignore_null": True, "attempt_sorting": False, "explanation": "Removed nulls from dataframes"},
        {"ignore_null": False, "attempt_sorting": True, "explanation": "Sorted dataframes"},
        {"ignore_null": True, "attempt_sorting": True, "explanation": "Removed nulls and sorted dataframes"},
    ]

    best_score = _ResultScore(equality_rank=4, error_count=2**31, warn_count=2**31)
    best_result = ComparisonResult()
    best_strategy = ""

    for cfg in configs:
        c = SqlResultComparator(
            ignore_null=cfg["ignore_null"],
            ignore_duplicates=False,
            attempt_sorting=cfg["attempt_sorting"],
            model=model,
            col_norm=col_norm,
        )
        r = c.compare(col_norm.gold.copy(), col_norm.pred.copy())

        if r.result == Equality.EQUAL:
            logger.info("Comparison equal with strategy: %s", cfg["explanation"])
            return r

        score = _res_to_score(r)
        if score < best_score:
            best_score = score
            best_result = r
            best_strategy = cfg["explanation"]

    logger.info("Best comparison strategy: %s -> %s", best_strategy, best_result.result)
    return best_result
