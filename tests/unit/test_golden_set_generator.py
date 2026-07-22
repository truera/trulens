"""Tests for the golden set generator benchmark utility."""

import json
import os
import tempfile
from typing import List, Optional, Tuple
import unittest

import pandas as pd
import pytest

try:
    from trulens.benchmark.golden_set_generator import GoldenSetGenerator
    from trulens.benchmark.golden_set_generator import GoldenSetSample
except ImportError:
    pytest.skip("trulens-benchmark not installed", allow_module_level=True)


class _StubSession:
    """Duck-typed TruSession exposing only what the generator uses."""

    def __init__(self, records_df: pd.DataFrame, feedback_names: List[str]):
        self.records_df = records_df
        self.feedback_names = feedback_names
        self.saved_datasets: List[Tuple[str, pd.DataFrame, Optional[dict]]] = []

    def get_records_and_feedback(
        self,
        app_name: Optional[str] = None,
        app_version: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, List[str]]:
        df = self.records_df
        if app_name is not None:
            df = df[df["app_name"] == app_name]
        if app_version is not None:
            df = df[df["app_version"] == app_version]
        if limit is not None:
            df = df.head(limit)
        return df.copy(), list(self.feedback_names)

    def add_ground_truth_to_dataset(
        self,
        dataset_name: str,
        ground_truth_df: pd.DataFrame,
        dataset_metadata: Optional[dict] = None,
    ) -> None:
        self.saved_datasets.append((
            dataset_name,
            ground_truth_df,
            dataset_metadata,
        ))


def _records_df(scores: List[Optional[float]]) -> pd.DataFrame:
    """Build a records DataFrame like `get_records_and_feedback` returns."""
    n = len(scores)
    return pd.DataFrame({
        "record_id": [f"record_{i}" for i in range(n)],
        "app_name": ["my_rag_app"] * n,
        "app_version": ["v1"] * n,
        "input": [json.dumps(f"question {i}") for i in range(n)],
        "output": [f"answer {i}" for i in range(n)],
        "ts": pd.to_datetime([
            pd.Timestamp("2026-05-01") + pd.Timedelta(days=i) for i in range(n)
        ]),
        "relevance": scores,
    })


def _generator(
    scores: List[Optional[float]], seed: int = 7
) -> Tuple[GoldenSetGenerator, _StubSession]:
    session = _StubSession(_records_df(scores), feedback_names=["relevance"])
    return GoldenSetGenerator(session, seed=seed), session


class TestGoldenSetSampling(unittest.TestCase):
    """Sampling strategies and filters."""

    def test_random_sample_has_golden_set_shape(self) -> None:
        generator, _ = _generator([0.1, 0.4, 0.6, 0.9, 0.5, 0.2])

        sample = generator.sample(n=3)

        self.assertEqual(len(sample), 3)
        golden = sample.to_df(include_provenance=False)
        self.assertEqual(
            list(golden.columns),
            ["query", "expected_response", "expected_score"],
        )
        self.assertTrue(golden["expected_score"].isna().all())
        # JSON-encoded inputs are unwrapped to plain text.
        self.assertTrue(golden["query"].str.startswith("question ").all())

    def test_random_sample_is_reproducible_with_seed(self) -> None:
        scores = [0.1, 0.4, 0.6, 0.9, 0.5, 0.2, 0.8, 0.3]
        generator_a, _ = _generator(scores, seed=42)
        generator_b, _ = _generator(scores, seed=42)

        ids_a = generator_a.sample(n=4).to_df()["record_id"].tolist()
        ids_b = generator_b.sample(n=4).to_df()["record_id"].tolist()

        self.assertEqual(ids_a, ids_b)

    def test_requesting_more_than_available_returns_all(self) -> None:
        generator, _ = _generator([0.1, 0.9])

        with self.assertLogs(
            "trulens.benchmark.golden_set_generator", level="WARNING"
        ):
            sample = generator.sample(n=10)

        self.assertEqual(len(sample), 2)

    def test_stratified_takes_equal_bucket_samples(self) -> None:
        scores = [0.1, 0.2, 0.15, 0.05, 0.5, 0.45, 0.55, 0.6, 0.9, 0.85, 0.95]
        generator, _ = _generator(scores)

        sample = generator.sample(
            n=6, strategy="stratified", feedback_name="relevance"
        )

        picked = sample.to_df()["feedback_score"]
        self.assertEqual(len(picked), 6)
        self.assertEqual((picked < 1 / 3).sum(), 2)
        self.assertEqual(((picked >= 1 / 3) & (picked < 2 / 3)).sum(), 2)
        self.assertEqual((picked >= 2 / 3).sum(), 2)

    def test_stratified_backfills_sparse_buckets(self) -> None:
        # Only one low-score record; the shortfall comes from other buckets.
        scores = [0.1, 0.5, 0.55, 0.6, 0.9, 0.85, 0.95, 0.8]
        generator, _ = _generator(scores)

        sample = generator.sample(
            n=6, strategy="stratified", feedback_name="relevance"
        )

        picked = sample.to_df()["feedback_score"]
        self.assertEqual(len(picked), 6)
        self.assertEqual((picked < 1 / 3).sum(), 1)

    def test_uncertainty_picks_scores_nearest_boundary(self) -> None:
        scores = [0.05, 0.48, 0.52, 0.95, 0.4, 0.99]
        generator, _ = _generator(scores)

        sample = generator.sample(
            n=3, strategy="uncertainty", feedback_name="relevance"
        )

        picked = sorted(sample.to_df()["feedback_score"])
        self.assertEqual(picked, [0.4, 0.48, 0.52])

    def test_uncertainty_respects_custom_boundary(self) -> None:
        scores = [0.05, 0.48, 0.52, 0.95, 0.4, 0.99]
        generator, _ = _generator(scores)

        sample = generator.sample(
            n=2,
            strategy="uncertainty",
            feedback_name="relevance",
            decision_boundary=1.0,
        )

        picked = sorted(sample.to_df()["feedback_score"])
        self.assertEqual(picked, [0.95, 0.99])

    def test_score_range_filter(self) -> None:
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        generator, _ = _generator(scores)

        sample = generator.sample(
            n=10, feedback_name="relevance", min_score=0.3, max_score=0.7
        )

        picked = sorted(sample.to_df()["feedback_score"])
        self.assertEqual(picked, [0.3, 0.5, 0.7])

    def test_date_range_filter(self) -> None:
        generator, _ = _generator([0.1, 0.3, 0.5, 0.7, 0.9])

        sample = generator.sample(
            n=10,
            start_time="2026-05-02",
            end_time="2026-05-04",
        )

        ids = sorted(sample.to_df()["record_id"])
        self.assertEqual(ids, ["record_1", "record_2", "record_3"])

    def test_unscored_records_dropped_for_score_strategies(self) -> None:
        scores = [0.1, None, 0.5, None, 0.9]
        generator, _ = _generator(scores)

        sample = generator.sample(
            n=10, strategy="uncertainty", feedback_name="relevance"
        )

        self.assertEqual(len(sample), 3)

    def test_records_without_input_are_dropped(self) -> None:
        df = _records_df([0.1, 0.5, 0.9])
        df.loc[1, "input"] = ""
        session = _StubSession(df, feedback_names=["relevance"])
        generator = GoldenSetGenerator(session, seed=7)

        sample = generator.sample(n=10)

        self.assertEqual(len(sample), 2)

    def test_non_string_output_is_json_encoded(self) -> None:
        df = _records_df([0.5])
        df["output"] = [{"answer": "42"}]
        session = _StubSession(df, feedback_names=["relevance"])
        generator = GoldenSetGenerator(session, seed=7)

        sample = generator.sample(n=1)

        self.assertEqual(
            sample.to_df()["expected_response"].iloc[0], '{"answer": "42"}'
        )

    def test_no_matching_records_returns_empty_sample(self) -> None:
        generator, _ = _generator([0.5])

        with self.assertLogs(
            "trulens.benchmark.golden_set_generator", level="WARNING"
        ):
            sample = generator.sample(n=5, start_time="2030-01-01")

        self.assertEqual(len(sample), 0)

    def test_invalid_arguments_raise(self) -> None:
        generator, _ = _generator([0.5])

        with self.assertRaises(ValueError):
            generator.sample(n=0)
        with self.assertRaises(ValueError):
            generator.sample(n=1, strategy="nonsense")
        with self.assertRaises(ValueError):
            generator.sample(n=1, strategy="stratified")
        with self.assertRaises(ValueError):
            generator.sample(n=1, min_score=0.5)
        with self.assertRaises(ValueError):
            generator.sample(
                n=1, strategy="uncertainty", feedback_name="unknown"
            )


class TestGoldenSetExportAndAnnotations(unittest.TestCase):
    """Export formats and the annotation round trip."""

    def _sample(self) -> Tuple[GoldenSetSample, GoldenSetGenerator]:
        generator, _ = _generator([0.1, 0.5, 0.9])
        return generator.sample(n=3), generator

    def test_to_list_matches_ground_truth_agreement_format(self) -> None:
        sample, _ = self._sample()

        golden_set = sample.to_list()

        self.assertEqual(len(golden_set), 3)
        for entry in golden_set:
            self.assertEqual(
                set(entry),
                {"query", "expected_response", "expected_score"},
            )
            self.assertIsNone(entry["expected_score"])

    def test_csv_annotation_round_trip(self) -> None:
        sample, generator = self._sample()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "golden_set.csv")
            sample.to_csv(path)

            annotated = pd.read_csv(path)
            annotated["expected_score"] = [1.0, 0.5, 0.0]
            annotated_path = os.path.join(tmp, "annotated.csv")
            annotated.to_csv(annotated_path, index=False)

            loaded = generator.load_annotations(annotated_path)

        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded["expected_score"].tolist(), [1.0, 0.5, 0.0])
        self.assertEqual(loaded["meta"].iloc[0]["expected_score"], 1.0)
        self.assertIn("record_id", loaded["meta"].iloc[0])

    def test_json_annotation_round_trip(self) -> None:
        sample, generator = self._sample()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "golden_set.json")
            sample.to_json(path)

            with open(path) as f:
                rows = json.load(f)
            for i, row in enumerate(rows):
                row["expected_score"] = i / 2
            annotated_path = os.path.join(tmp, "annotated.json")
            with open(annotated_path, "w") as f:
                json.dump(rows, f)

            loaded = generator.load_annotations(annotated_path)

        self.assertEqual(loaded["expected_score"].tolist(), [0.0, 0.5, 1.0])

    def test_load_annotations_accepts_list_of_dicts(self) -> None:
        _, generator = self._sample()

        loaded = generator.load_annotations([
            {"query": "q1", "expected_response": "a1", "expected_score": 0.25}
        ])

        self.assertEqual(loaded["meta"].iloc[0], {"expected_score": 0.25})

    def test_load_annotations_validation_errors(self) -> None:
        _, generator = self._sample()

        with self.assertRaises(ValueError):
            generator.load_annotations([{"expected_score": 0.5}])
        with self.assertRaises(ValueError):
            generator.load_annotations([{"query": "", "expected_score": 0.5}])
        with self.assertRaises(ValueError):
            generator.load_annotations([
                {"query": "q1", "expected_score": None}
            ])
        with self.assertRaises(ValueError):
            generator.load_annotations([{"query": "q1", "expected_score": 1.5}])

    def test_load_annotations_optional_scores(self) -> None:
        _, generator = self._sample()

        loaded = generator.load_annotations(
            [{"query": "q1"}], require_scores=False
        )

        self.assertTrue(pd.isna(loaded["expected_score"].iloc[0]))
        self.assertEqual(loaded["meta"].iloc[0], {})

    def test_save_golden_set_persists_via_session(self) -> None:
        generator, session = _generator([0.1, 0.5, 0.9])
        sample = generator.sample(n=3)

        annotated = sample.to_df()
        annotated["expected_score"] = [0.0, 0.5, 1.0]
        count = generator.save_golden_set(
            "my_golden_set",
            generator.load_annotations(annotated),
            dataset_metadata={"source": "test"},
        )

        self.assertEqual(count, 3)
        self.assertEqual(len(session.saved_datasets), 1)
        name, df, metadata = session.saved_datasets[0]
        self.assertEqual(name, "my_golden_set")
        self.assertEqual(metadata, {"source": "test"})
        self.assertIn("meta", df.columns)
        self.assertEqual(df["meta"].iloc[0]["expected_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
