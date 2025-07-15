#!/usr/bin/env python3

from abc import ABC
from abc import abstractmethod
import argparse
import copy
from dataclasses import dataclass
from dataclasses import field
import logging
import math
from typing import Any, Iterable

import pandas as pd
from pandas import DataFrame
from pandas import Series
from pandas import read_csv
import regex as re
import scipy

NO_NEED_TO_BE_QUOTED_REGEX = re.compile(r"^[\p{Lu}_]+$")

# heuristic regexp for automatically removing columns with scores or
# other values likely to correlate with the main score in a trivial manner;
# basically looking for columns with "score" and other words suggesting
# some sort of score
AUTOSKIP_COLUMNS_REGEX = re.compile(
    r"score|precision|recall|(false|true)[-_ ](negative|positive)|llm[-_ ](as[-_ ])?judge",
    flags=re.IGNORECASE,
)

QUOTE_CHARACTER = '"'


def quote_if_needed(val: str) -> str:
    if re.match(NO_NEED_TO_BE_QUOTED_REGEX, val):
        return val

    return (
        QUOTE_CHARACTER
        + val.replace(QUOTE_CHARACTER, QUOTE_CHARACTER + QUOTE_CHARACTER)
        + QUOTE_CHARACTER
    )


def shorter_string_preference(s: str) -> float:
    # sigmoid on string length
    return scipy.special.expit(len(s))


def should_be_autoskipped(column: str) -> bool:
    return bool(AUTOSKIP_COLUMNS_REGEX.search(column))


def get_autoskipped_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if should_be_autoskipped(c)]


@dataclass
class HotspotsConfig:
    score_column: str
    skip_columns: list[str] = field(default_factory=list)
    more_skipped_columns: list[str] = field(default_factory=list)
    min_occurrences: int = 5
    max_features: int = 1000
    num_rounds: int = 20
    higher_is_better: bool = True


class Feature(ABC):
    def priority(self) -> float:
        """Priority for features, the lower value, the more it should be preferred"""
        return 1.0

    @abstractmethod
    def description(self) -> str:
        """Human-readable description"""
        pass


@dataclass(frozen=True)
class TokenFeature(Feature):
    name: str
    value: str

    def priority(self) -> float:
        # We prefer shorter values, similarly as for EqualsFeature, but here it's less important.
        return 1.0 + shorter_string_preference(self.value)

    def __str__(self) -> str:
        return f"{self.name}:{self.value}"

    def description(self) -> str:
        return f"{quote_if_needed(self.name)} contains {quote_if_needed(self.value)}"


@dataclass(frozen=True)
class EqualsFeature(Feature):
    name: str
    value: str

    def priority(self) -> float:
        # We prefer shorter values: imagine two columns perfectly correlating with each other,
        # usually, it's better to list the shorter value.
        return shorter_string_preference(self.value)

    def __str__(self) -> str:
        return f"{self.name}={self.value}"

    def description(self) -> str:
        return (
            f"{quote_if_needed(self.name)} equals {quote_if_needed(self.value)}"
        )


@dataclass(frozen=True)
class NumericalFeatureBase(Feature, ABC):
    name: str
    threshold: float
    greater_or_equal: bool

    @classmethod
    def generate_features(
        cls, name: str, thresholds: list[float], value: float
    ) -> set[Feature]:
        n = cls._get_number(value)
        return set(cls.generate_feature(name, th, n) for th in thresholds)

    @classmethod
    def generate_feature(cls, name: str, threshold: float, n: float) -> Feature:
        return cls(
            name=name, threshold=threshold, greater_or_equal=(n >= threshold)
        )

    @classmethod
    @abstractmethod
    def _get_number(cls, v: Any) -> float:
        pass

    @abstractmethod
    def _format_threshold(self) -> str:
        pass

    @classmethod
    def get_thresholds(cls, col: Series) -> list[float]:  # type: ignore[type-arg]
        d = col.apply(lambda v: cls._get_number(v))

        m = d.median()
        ths = [m]

        q1 = d.quantile(0.25)
        if q1 < m:
            ths.append(q1)

        q3 = d.quantile(0.75)
        if q1 > m:
            ths.append(q3)

        return ths

    @abstractmethod
    def _name_description(self) -> str:
        pass

    def _relation_description(self) -> str:
        if self.greater_or_equal:
            return "greater than or equal to"
        else:
            return "smaller than"

    def description(self) -> str:
        return f"{self._name_description()} {self._relation_description()} {self._format_threshold()}"


def is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


@dataclass(frozen=True)
class NumericalFeature(NumericalFeatureBase):
    def priority(self) -> float:
        return 2.0

    @classmethod
    def is_applicable(cls, value: Any) -> bool:
        return is_number(value)

    @classmethod
    def _get_number(cls, value: Any) -> float:
        return float(value)

    def _format_threshold(self) -> str:
        return str(self.threshold)

    def __str__(self) -> str:
        s = ">=" if self.greater_or_equal else "<"
        return f"{self.name}{s}{self.threshold}"

    def _name_description(self) -> str:
        return quote_if_needed(self.name)


@dataclass(frozen=True)
class LengthFeature(NumericalFeatureBase):
    def priority(self) -> float:
        return 3.0

    @classmethod
    def is_applicable(cls, value: Any) -> bool:
        return True

    @classmethod
    def _get_number(cls, value: Any) -> float:
        return float(len(str(value)))

    def _format_threshold(self) -> str:
        return str(round(self.threshold))

    def __str__(self) -> str:
        s = ">=" if self.greater_or_equal else "<"
        return f"len({self.name}){s}{self._format_threshold()}"

    def _name_description(self) -> str:
        return f"length of {quote_if_needed(self.name)}"


@dataclass
class FeatureStats:
    p_value: float
    average_score: float
    num_occurrences: int


def get_features_for_value(
    config: HotspotsConfig, column_name: str, value: Any
) -> set[Feature]:
    return set(
        TokenFeature(name=column_name, value=t)
        for t in re.findall(r"[0-9\p{L}]+", str(value))
    ) | set([EqualsFeature(name=column_name, value=str(value))])


def get_features_for_sample(
    config: HotspotsConfig,
    general_num_stats: list[tuple[type[Feature], str, list[float]]],
    sample: Series,  # type: ignore[type-arg]
) -> set[Feature]:
    num_features = set()
    for feature_type, col, thresholds in general_num_stats:
        v = sample[col]
        if feature_type.is_applicable(v):  # type: ignore[attr-defined]
            num_features |= feature_type.generate_features(col, thresholds, v)  # type: ignore[attr-defined]

    return num_features | set.union(
        *(
            get_features_for_value(config, col, sample[col])
            for col in sample.index
            if col != config.score_column
        )
    )


def get_inverted_index_for_features(
    config: HotspotsConfig,
    general_num_stats: list[tuple[type[Feature], str, list[float]]],
    df: DataFrame,
) -> dict[Feature, list[int]]:
    inverted_index: dict[Feature, list[int]] = {}

    for index, row in df.iterrows():
        assert isinstance(index, int)

        features = get_features_for_sample(config, general_num_stats, row)

        for feature in features:
            if feature in inverted_index:
                inverted_index[feature].append(index)
            else:
                inverted_index[feature] = [index]

    return inverted_index


def prune_inverted_index(
    min_occurrences: int,
    num_samples: int,
    inverted_index: dict[Feature, list[int]],
) -> None:
    for feature, occurrences in list(inverted_index.items()):
        num_occurrences = len(occurrences)
        if (
            num_occurrences < min_occurrences
            or num_occurrences > num_samples - min_occurrences
        ):
            del inverted_index[feature]


def z_to_p_value(z: float) -> float:
    p_value: float = scipy.stats.norm.sf(z)
    return p_value


def utest_z(crank: float, poss: int, negs: int) -> float:
    # Handle edge cases where Mann-Whitney U test is undefined
    if poss == 0 or negs == 0:
        # When feature occurs in ALL samples (negs=0) or NO samples (poss=0),
        # there's no valid comparison group, making the test undefined.
        # Return 0.0 as a practical default to ensure such features are
        # deprioritized (p-value = 0.5) in hotspot ranking.
        return 0.0

    minus_r = poss * (poss + 1.0) / 2.0
    mean = (poss * negs) / 2.0
    sigma = math.sqrt(poss * negs * (poss + negs + 1) / 12.0)
    u = crank - minus_r
    z = (u - mean) / sigma
    return z


def get_feature_stat(
    ranks: Series,  # type: ignore[type-arg]
    scores: Series,  # type: ignore[type-arg]
    feature: Feature,
    occurrences: list[int],
) -> FeatureStats:
    crank = sum((ranks[idx] for idx in occurrences), 0.0)

    num_occurrences = len(occurrences)
    num_total = scores.shape[0]

    z = utest_z(crank, num_occurrences, num_total - num_occurrences)

    return FeatureStats(
        p_value=z_to_p_value(z),
        # Prevent division by zero: average of empty list is mathematically undefined.
        # Return 0.0 to represent "no signal" when feature has no occurrences.
        # This maintains consistency with other edge case handling and prevents NaN propagation.
        average_score=sum((scores[idx] for idx in occurrences), 0.0)
        / num_occurrences
        if num_occurrences > 0
        else 0.0,
        num_occurrences=num_occurrences,
    )


def prepare_inverted_index(
    config: HotspotsConfig,
    general_num_stats: list[tuple[type[Feature], str, list[float]]],
    df: DataFrame,
) -> dict[Feature, list[int]]:
    index = get_inverted_index_for_features(config, general_num_stats, df)
    prune_inverted_index(config.min_occurrences, df.shape[0], index)

    return index


def get_features_with_stats(
    ranks: Series,  # type: ignore[type-arg]
    scores: Series,  # type: ignore[type-arg]
    index: dict[Feature, list[int]],
    fs: Iterable[Feature],
) -> list[tuple[Feature, FeatureStats]]:
    return sorted(
        [(f, get_feature_stat(ranks, scores, f, index[f])) for f in fs],
        key=lambda e: (e[1].p_value, e[0].priority(), str(e[0])),
    )


def get_skipped_columns(
    config: HotspotsConfig, columns: list[str]
) -> list[str]:
    if config.skip_columns:
        skip_columns = config.skip_columns
    else:
        skip_columns = [
            c
            for c in get_autoskipped_columns(columns)
            if c != config.score_column
        ]
        logging.warning(
            f"automatically skipping the following columns: {', '.join(skip_columns)}"
        )

    if config.more_skipped_columns:
        skip_columns += config.more_skipped_columns

    return skip_columns


def initial_round(
    config: HotspotsConfig, df: DataFrame
) -> tuple[
    DataFrame, dict[Feature, list[int]], list[tuple[Feature, FeatureStats]]
]:
    skip_columns = get_skipped_columns(config, list(df.columns))

    if skip_columns:
        df = df.drop(skip_columns, axis=1)

    general_num_stats: list[tuple[type[Feature], str, list[float]]] = []

    for col in df.columns:
        if col != config.score_column:
            ths = LengthFeature.get_thresholds(df[col])
            general_num_stats.append((LengthFeature, col, ths))

            if pd.api.types.is_numeric_dtype(df[col]):
                ths = NumericalFeature.get_thresholds(df[col])
                general_num_stats.append((NumericalFeature, col, ths))

    index = prepare_inverted_index(config, general_num_stats, df)

    scores = df[config.score_column]
    ranks = df[config.score_column].rank(
        method="average", ascending=not config.higher_is_better
    )
    fs = (get_features_with_stats(ranks, scores, index, index.keys()))[
        : config.max_features
    ]

    return (df, index, fs)


def delta(num_total: int, average_score: float, stats: FeatureStats) -> float:
    # Handle degenerate case where feature appears in ALL samples
    if num_total == stats.num_occurrences:
        # Mathematically undefined: no "without feature" group exists for comparison.
        # Return 0.0 to indicate no discriminating power (not a useful hotspot).
        # This prevents division by zero in: (total_score - total_poss) / 0
        return 0.0

    total_score = num_total * average_score
    total_poss = stats.num_occurrences * stats.average_score

    neg_average_score = (total_score - total_poss) / (
        num_total - stats.num_occurrences
    )

    return stats.average_score - neg_average_score


def opportunity(
    num_total: int, average_score: float, stats: FeatureStats
) -> float:
    # Handle degenerate case where feature appears in ALL samples
    if num_total == stats.num_occurrences:
        # Mathematically undefined: no "without feature" group exists for comparison.
        # Return 0.0 to indicate no improvement opportunity.
        # This prevents division by zero in: (total_score - total_poss) / 0
        return 0.0

    total_score = num_total * average_score
    total_poss = stats.num_occurrences * stats.average_score

    neg_average_score = (total_score - total_poss) / (
        num_total - stats.num_occurrences
    )

    return neg_average_score - average_score


def is_feature_unwanted(
    index: dict[Feature, list[int]], fs_listed: set[Feature], fs: Feature
) -> bool:
    for ofs in fs_listed:
        if index[fs] == index[ofs]:
            return True

    return False


def do_round(
    scores: Series,  # type: ignore[type-arg]
    index: dict[Feature, list[int]],
    fs: list[tuple[Feature, FeatureStats]],
    higher_is_better: bool,
) -> tuple[Series, list[tuple[Feature, FeatureStats]]]:  # type: ignore[type-arg]
    top_fs = fs[0]
    fs_rest = fs[1:]

    num_total = scores.shape[0]
    average_score = scores.mean()
    d = delta(num_total, average_score, top_fs[1])

    for ix in index[top_fs[0]]:
        scores.at[ix] = scores[ix] - d

    ranks = scores.rank(method="average", ascending=not higher_is_better)
    fs_rest = get_features_with_stats(
        ranks, scores, index, (f[0] for f in fs_rest)
    )

    return (scores, fs_rest)


def hotspots(
    config: HotspotsConfig, df: DataFrame
) -> tuple[DataFrame, float, list[tuple[Feature, FeatureStats, list[int]]]]:
    """
    Return a list of hotspots for a given data frame with evaluation scores.

    This is the main function of this library.

    Parameters:
    config (HotspotConfig): config specifying which column contains the evaluation score and other options
    df (DataFrame): Pandas data frame with per-item evaluation scores

    Returns:
    (DataFrame, float, list[tuple[Feature, FeatureStats]]): a triple with the following values:
      - modified data frame
      - average score
      - list of hotspot features and their statistics
    """
    original_num_total = df.shape[0]
    df = df[df[config.score_column].notna()]
    num_total = df.shape[0]
    if num_total < original_num_total:
        logging.warning(
            f"{original_num_total - num_total} samples without a score, skipping them"
        )
    avg_score = df[config.score_column].mean()

    logging.info("starting initial round...")
    df, index, fs = initial_round(config, df)
    logging.info(f"initial round finished with {len(fs)} features")

    original_fs = {f: copy.deepcopy(s) for f, s in fs}

    scores = df[config.score_column]

    outs = []

    fs_listed = set()

    for round_ix in range(config.num_rounds):
        if not fs:
            logging.info("not enough eligible features, stopping...")
            break

        top_feature_name = fs[0][0]

        logging.info(f"ROUND {round_ix} WITH TOP FEATURE {top_feature_name}")

        outs.append((
            top_feature_name,
            original_fs[top_feature_name],
            index[top_feature_name],
        ))
        fs_listed.add(top_feature_name)

        scores, fs = do_round(scores, index, fs, config.higher_is_better)

        while fs and is_feature_unwanted(index, fs_listed, fs[0][0]):
            fs = fs[1:]

        if not fs:
            break

    return (df, avg_score, outs)


def to_dict_for_df(
    num_total: int, avg_score: float, hts: Feature, hts_stats: FeatureStats
) -> dict[str, Any]:
    return {
        "hotspot": hts.description(),
        "#": hts_stats.num_occurrences,
        "avg score": hts_stats.average_score,
        "deterioration": delta(num_total, avg_score, hts_stats),
        "opportunity": opportunity(num_total, avg_score, hts_stats),
        "p-value": hts_stats.p_value,
    }


def hotspots_dict_to_df(
    out_df: DataFrame,
    avg_score: float,
    htss: list[tuple[Feature, FeatureStats, list[int]]],
) -> DataFrame:
    num_total = out_df.shape[0]

    return DataFrame([
        to_dict_for_df(num_total, avg_score, hts, hts_stats)
        for hts, hts_stats, _ in htss
    ])


def hotspots_as_df(config: HotspotsConfig, df: DataFrame) -> DataFrame:
    out_df, avg_score, htss = hotspots(config, df)

    return hotspots_dict_to_df(out_df, avg_score, htss)


def format_feature_with_stat(
    num_total: int,
    average_score: float,
    feature: Feature,
    feature_stat: FeatureStats,
) -> str:
    return f"{feature}\t{feature_stat.num_occurrences}\t{feature_stat.average_score:0.8f}\t{delta(num_total, average_score, feature_stat):+.08f}\t{opportunity(num_total, average_score, feature_stat):+.08f}\t{feature_stat.p_value:.020f}"


def parse_args() -> tuple[str, HotspotsConfig]:
    parser = argparse.ArgumentParser(
        description="TruLens Hotspots: Tool for listing features correlating with worsening of your evaluation scores"
    )

    parser.add_argument(
        "csv_filename", type=str, help="CSV file with evaluation data"
    )
    parser.add_argument(
        "--score_column",
        type=str,
        required=True,
        help="Name of the column with evaluation scores",
    )
    parser.add_argument(
        "--skip_columns",
        type=str,
        nargs="+",
        help="Columns to be disregarded, overriding the list of columns inferred automatically using heuristics",
    )
    parser.add_argument(
        "--more_skipped_columns",
        type=str,
        nargs="+",
        help="Columns to be disregarded, adding to the list of columns inferred automatically using heuristics",
    )
    parser.add_argument(
        "--lower_is_better",
        action="store_true",
        help="The-lower-the-better score, e.g. MAE, RMSE, WER",
    )

    args = parser.parse_args()
    return (
        args.csv_filename,
        HotspotsConfig(
            score_column=args.score_column,
            skip_columns=args.skip_columns,
            more_skipped_columns=args.more_skipped_columns,
            higher_is_better=not args.lower_is_better,
        ),
    )


def main():
    logging.basicConfig(level=logging.INFO)

    (csv_filename, config) = parse_args()

    df = read_csv(csv_filename)

    df, avg_score, out = hotspots(config, df)

    num_total = df.shape[0]

    for item in out:
        print(format_feature_with_stat(num_total, avg_score, item[0], item[1]))


if __name__ == "__main__":
    main()
