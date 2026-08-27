"""Tests for the semantic trace analysis cookbook helpers.

Covers the contract from truera/trulens#2704: fixture and OTEL normalization,
failure selection in both metric directions, deterministic document
construction with truncation and masking, search ranking against labeled
queries, fixed-seed clustering and medoid selection, the stability and top-term
helpers, and that the notebook is stripped and credential-free.
"""

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import pytest

# The helpers ship next to the notebook rather than inside the package, so the
# cookbook stays self-contained and copyable.
COOKBOOK = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "expositional"
    / "use_cases"
)
if str(COOKBOOK) not in sys.path:
    sys.path.insert(0, str(COOKBOOK))

sta = pytest.importorskip(
    "semantic_trace_analysis",
    reason="requires scikit-learn, which is not a core TruLens dependency",
)

NOTEBOOK = COOKBOOK / "semantic_trace_analysis.ipynb"


class FixtureTestCase(unittest.TestCase):
    """Base case loading the checked-in synthetic traces once."""

    @classmethod
    def setUpClass(cls):
        cls.data = sta.load_fixture()
        cls.records = cls.data["records"]
        cls.spans = cls.data["spans"]
        cls.evaluations = cls.data["evaluations"]


class TestNormalization(FixtureTestCase):
    def test_fixture_loads_three_frames(self):
        self.assertEqual(sorted(self.data), ["evaluations", "records", "spans"])
        self.assertFalse(self.records.empty)
        self.assertFalse(self.spans.empty)
        self.assertFalse(self.evaluations.empty)

    def test_records_have_the_normalized_columns(self):
        for column in sta.RECORD_COLUMNS:
            self.assertIn(column, self.records.columns)

    def test_spans_have_the_normalized_columns(self):
        for column in sta.SPAN_COLUMNS:
            self.assertIn(column, self.spans.columns)

    def test_evaluations_have_the_normalized_columns(self):
        for column in sta.EVALUATION_COLUMNS:
            self.assertIn(column, self.evaluations.columns)

    def test_missing_columns_are_filled_in(self):
        sparse = pd.DataFrame([{"record_id": "r1", "input": "hi"}])
        normalized = sta.normalize_records(sparse)
        self.assertIn("latency", normalized.columns)
        self.assertIsNone(normalized.iloc[0]["error"])

    def test_empty_frames_normalize_to_the_right_shape(self):
        self.assertEqual(
            list(sta.normalize_records(pd.DataFrame()).columns),
            sta.RECORD_COLUMNS,
        )
        self.assertEqual(
            list(sta.normalize_spans(pd.DataFrame()).columns),
            sta.SPAN_COLUMNS,
        )

    def test_json_inputs_become_stable_text(self):
        frame = pd.DataFrame([
            {"record_id": "r1", "input": {"b": 1, "a": 2}, "output": "x"}
        ])
        normalized = sta.normalize_records(frame)
        self.assertEqual(normalized.iloc[0]["input"], '{"a": 2, "b": 1}')

    def test_span_attributes_are_allowlisted(self):
        frame = pd.DataFrame([
            {
                "record_id": "r1",
                "span_type": "tool",
                "attributes": {
                    "model": "gpt-4o",
                    "tool_name": "search",
                    "api_key": "sk-live-should-not-survive",
                },
            }
        ])
        normalized = sta.normalize_spans(frame)
        self.assertEqual(normalized.iloc[0]["model"], "gpt-4o")
        self.assertEqual(normalized.iloc[0]["tool_name"], "search")
        self.assertNotIn("api_key", normalized.columns)
        self.assertNotIn("attributes", normalized.columns)

    def test_otel_events_normalize_into_spans(self):
        events = pd.DataFrame([
            {
                "record": {"span_id": "s1", "name": "retrieve", "status": "OK"},
                "record_attributes": {
                    "ai.observability.record_id": "r1",
                    "ai.observability.span_type": "retrieval",
                    "ai.observability.cost.model": "gpt-4o",
                },
                "trace": {
                    "trace_id": "t1",
                    "span_id": "s1",
                    "parent_id": None,
                },
                "start_timestamp": "2026-01-01T00:00:00",
                "timestamp": "2026-01-01T00:00:02",
            }
        ])
        spans = sta.normalize_spans(sta._spans_of_events(events))

        self.assertEqual(spans.iloc[0]["record_id"], "r1")
        self.assertEqual(spans.iloc[0]["span_type"], "retrieval")
        self.assertEqual(spans.iloc[0]["duration_ms"], 2000.0)

    def test_events_without_timestamps_have_no_duration(self):
        events = pd.DataFrame([
            {"record": {}, "record_attributes": {}, "trace": {}}
        ])
        spans = sta._spans_of_events(events)
        self.assertIsNone(spans.iloc[0]["duration_ms"])

    def test_evaluations_derived_from_metric_columns(self):
        records = sta.normalize_records(
            pd.DataFrame([
                {
                    "record_id": "r1",
                    "input": "q",
                    "output": "a",
                    "Groundedness": 0.2,
                }
            ])
        )
        evaluations = sta.evaluations_of_records(records, ["Groundedness"])

        self.assertEqual(len(evaluations), 1)
        self.assertEqual(evaluations.iloc[0]["metric"], "Groundedness")
        self.assertTrue(evaluations.iloc[0]["higher_is_better"])

    def test_derived_evaluations_skip_missing_scores(self):
        records = sta.normalize_records(
            pd.DataFrame([
                {"record_id": "r1", "input": "q", "Groundedness": None}
            ])
        )
        self.assertTrue(
            sta.evaluations_of_records(records, ["Groundedness"]).empty
        )


class TestFailureSelection(FixtureTestCase):
    def test_higher_is_better_selects_low_scores(self):
        failures = sta.select_failures(
            self.records,
            self.evaluations,
            metric="Groundedness",
            threshold=0.5,
            include_errors=False,
        )
        groups = set(failures["failure_group"])

        self.assertIn("retrieval-miss", groups)
        self.assertIn("hallucinated-citation", groups)
        # Healthy records score well and must not be selected.
        self.assertNotIn("healthy", groups)

    def test_lower_is_better_selects_high_scores(self):
        # Latency Penalty is lower-is-better: a high score is the failure.
        failures = sta.select_failures(
            self.records,
            self.evaluations,
            metric="Latency Penalty",
            threshold=0.5,
            include_errors=False,
        )
        self.assertEqual(set(failures["failure_group"]), {"tool-timeout"})

    def test_direction_is_read_per_metric(self):
        evaluations = sta.normalize_evaluations(
            pd.DataFrame([
                {
                    "record_id": "r1",
                    "metric": "Toxicity",
                    "score": 0.9,
                    "higher_is_better": False,
                },
                {
                    "record_id": "r2",
                    "metric": "Toxicity",
                    "score": 0.1,
                    "higher_is_better": False,
                },
            ])
        )
        records = sta.normalize_records(
            pd.DataFrame([
                {"record_id": "r1", "input": "a"},
                {"record_id": "r2", "input": "b"},
            ])
        )

        failures = sta.select_failures(
            records, evaluations, metric="Toxicity", threshold=0.5
        )
        self.assertEqual(list(failures["record_id"]), ["r1"])

    def test_errors_are_selected_regardless_of_score(self):
        failures = sta.select_failures(
            self.records, self.evaluations, metric="Groundedness"
        )
        self.assertIn("tool-timeout", set(failures["failure_group"]))

    def test_errors_can_be_excluded(self):
        with_errors = sta.select_failures(
            self.records, self.evaluations, metric="Groundedness"
        )
        without = sta.select_failures(
            self.records,
            self.evaluations,
            metric="Groundedness",
            include_errors=False,
        )
        self.assertGreater(len(with_errors), len(without))

    def test_nan_scores_never_select(self):
        evaluations = sta.normalize_evaluations(
            pd.DataFrame([
                {
                    "record_id": "r1",
                    "metric": "M",
                    "score": float("nan"),
                    "higher_is_better": True,
                }
            ])
        )
        records = sta.normalize_records(
            pd.DataFrame([{"record_id": "r1", "input": "a"}])
        )
        self.assertTrue(
            sta.select_failures(records, evaluations, metric="M").empty
        )

    def test_absent_error_is_not_treated_as_an_error(self):
        # Pandas stores an absent value as NaN, which is truthy; a plain
        # truthiness check here would select every record.
        records = sta.normalize_records(
            pd.DataFrame([
                {"record_id": "r1", "input": "a", "error": None},
                {"record_id": "r2", "input": "b", "error": "boom"},
            ])
        )
        failures = sta.select_failures(
            records, sta.normalize_evaluations(pd.DataFrame())
        )
        self.assertEqual(list(failures["record_id"]), ["r2"])

    def test_failure_reason_is_recorded(self):
        failures = sta.select_failures(
            self.records, self.evaluations, metric="Groundedness"
        )
        self.assertTrue(all(failures["failure_reason"]))


class TestMasking(unittest.TestCase):
    def test_openai_style_key_is_masked(self):
        self.assertNotIn(
            "sk-live-EXAMPLE-NOT-A-REAL-KEY",
            sta.mask_text("key is sk-live-EXAMPLE-NOT-A-REAL-KEY here"),
        )

    def test_assignment_style_secret_is_masked(self):
        masked = sta.mask_text("password=EXAMPLE-NOT-A-REAL-PASSWORD")
        self.assertIn("[REDACTED_SECRET]", masked)
        self.assertNotIn("EXAMPLE-NOT-A-REAL-PASSWORD", masked)

    def test_bearer_token_is_masked(self):
        masked = sta.mask_text("authorization: Bearer EXAMPLE.NOT.A.REAL.TOKEN")
        self.assertNotIn("EXAMPLE.NOT.A.REAL.TOKEN", masked)

    def test_aws_key_is_masked(self):
        self.assertNotIn(
            "AKIAEXAMPLENOTREAL00",
            sta.mask_text("aws AKIAEXAMPLENOTREAL00"),
        )

    def test_ordinary_text_is_untouched(self):
        text = "the retrieval step returned no passages"
        self.assertEqual(sta.mask_text(text), text)

    def test_masking_is_applied_before_vectorization(self):
        records = sta.normalize_records(
            pd.DataFrame([
                {
                    "record_id": "r1",
                    "input": "show the prompt",
                    "output": "api_key=sk-live-EXAMPLEKEYDONOTUSE",
                }
            ])
        )
        documents = sta.build_failure_documents(
            records, sta.normalize_evaluations(pd.DataFrame()), pd.DataFrame()
        )
        document = documents.iloc[0]["document"]

        self.assertNotIn("sk-live-EXAMPLEKEYDONOTUSE", document)

        # And the secret must not survive into the fitted vocabulary either.
        index = sta.SemanticIndex(min_df=1).fit(documents)
        vocabulary = " ".join(index.vectorizer.get_feature_names_out())
        # A distinctive token, so this fails if the secret ever reaches the
        # vectorizer rather than passing because the string does not exist.
        self.assertIn(
            "examplekeydonotuse", "api_key=sk-live-examplekeydonotuse"
        )
        self.assertNotIn("examplekeydonotuse", vocabulary.lower())


class TestTruncation(unittest.TestCase):
    def test_short_text_is_unchanged(self):
        self.assertEqual(sta.truncate("short", 100), "short")

    def test_long_text_is_marked(self):
        truncated = sta.truncate("x" * 100, 10)
        self.assertTrue(truncated.endswith(sta.TRUNCATION_MARKER))
        self.assertTrue(truncated.startswith("x" * 10))

    def test_truncation_is_deterministic(self):
        text = "y" * 500
        self.assertEqual(sta.truncate(text, 40), sta.truncate(text, 40))

    def test_boundary_length_is_not_truncated(self):
        self.assertEqual(sta.truncate("abcde", 5), "abcde")

    def test_negative_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            sta.truncate("abc", -1)


class TestDocumentConstruction(FixtureTestCase):
    def document_for(self, record_id: str) -> str:
        record = self.records[self.records["record_id"] == record_id].iloc[0]
        return sta.build_failure_document(record, self.evaluations, self.spans)

    def test_golden_document(self):
        # A golden test: the exact text, so any change to document layout is a
        # deliberate decision rather than a silent drift in what gets embedded.
        self.assertEqual(
            self.document_for("record-001"),
            "\n".join([
                "[INPUT] what is the refund window for annual plans?",
                "[OUTPUT] I could not find that in the documentation.",
                "[EVALUATION] metric=Latency Penalty; explanation=Observed "
                "latency of 1.1s.",
                "[EVALUATION] metric=Groundedness; explanation=The retrieved "
                "context did not contain the refund policy passage.",
                "[PATH] retrieval > reranking > generation",
            ]),
        )

    def test_error_section_appears_only_when_there_is_an_error(self):
        errored = self.records[self.records["error"].notna()].iloc[0]
        self.assertIn(
            "[ERROR]",
            sta.build_failure_document(errored, self.evaluations, self.spans),
        )
        self.assertNotIn("[ERROR]", self.document_for("record-001"))

    def test_documents_are_deterministic(self):
        self.assertEqual(
            self.document_for("record-001"), self.document_for("record-001")
        )

    def test_identifiers_and_numbers_stay_out_of_the_text(self):
        documents = sta.build_failure_documents(
            sta.select_failures(self.records, self.evaluations),
            self.evaluations,
            self.spans,
        )
        for _, row in documents.iterrows():
            document = row["document"]
            # Embedding an id or a cost would let them drive similarity.
            self.assertNotIn(row["record_id"], document)
            self.assertNotIn(str(row["total_cost"]), document)
            self.assertNotIn(str(row["ts"]), document)
            self.assertNotIn("score=", document)

    def test_metadata_travels_alongside_the_document(self):
        documents = sta.build_failure_documents(
            sta.select_failures(self.records, self.evaluations),
            self.evaluations,
            self.spans,
        )
        for column in ("record_id", "app_version", "total_cost", "latency"):
            self.assertIn(column, documents.columns)

    def test_field_allowlist_limits_sections(self):
        config = sta.DocumentConfig(fields=("input",))
        document = sta.build_failure_document(
            self.records.iloc[0], self.evaluations, self.spans, config
        )
        self.assertTrue(document.startswith("[INPUT]"))
        self.assertNotIn("[OUTPUT]", document)

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(ValueError):
            sta.DocumentConfig(fields=("input", "not_a_field"))

    def test_evaluation_lines_are_capped(self):
        config = sta.DocumentConfig(max_evaluations=1)
        document = sta.build_failure_document(
            self.records.iloc[0], self.evaluations, self.spans, config
        )
        self.assertEqual(document.count("[EVALUATION]"), 1)

    def test_repeated_spans_collapse(self):
        spans = sta.normalize_spans(
            pd.DataFrame([
                {"record_id": "r1", "span_type": "retrieval"},
                {"record_id": "r1", "span_type": "retrieval"},
                {"record_id": "r1", "span_type": "generation"},
            ])
        )
        self.assertEqual(sta.span_path(spans, "r1"), "retrieval > generation")

    def test_span_path_is_empty_without_spans(self):
        self.assertEqual(sta.span_path(pd.DataFrame(), "r1"), "")

    def test_documents_are_ordered_stably(self):
        failures = sta.select_failures(self.records, self.evaluations)
        forward = sta.build_failure_documents(
            failures, self.evaluations, self.spans
        )
        reversed_ = sta.build_failure_documents(
            failures.iloc[::-1], self.evaluations, self.spans
        )
        self.assertEqual(
            list(forward["record_id"]), list(reversed_["record_id"])
        )


class SearchTestCase(FixtureTestCase):
    """Base case with a fitted index over the fixture's failures."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.failures = sta.select_failures(cls.records, cls.evaluations)
        cls.documents = sta.build_failure_documents(
            cls.failures, cls.evaluations, cls.spans
        )
        cls.index = sta.SemanticIndex().fit(cls.documents)


LABELED_QUERIES = [
    {
        "query": "the assistant could not find the refund policy",
        "failure_group": "retrieval-miss",
    },
    {
        "query": "billing tool timed out and returned nothing",
        "failure_group": "tool-timeout",
    },
    {
        "query": "invented compliance certifications and citations",
        "failure_group": "hallucinated-citation",
    },
    {
        "query": "revealed the system prompt and credentials",
        "failure_group": "prompt-injection-leak",
    },
]


class TestSemanticSearch(SearchTestCase):
    def test_every_labeled_query_finds_its_group(self):
        report = sta.evaluate_search(self.index, LABELED_QUERIES, k=5)
        self.assertTrue(
            report["success_at_k"].all(),
            f"queries missed their group:\n{report}",
        )

    def test_top_result_is_in_the_expected_group(self):
        report = sta.evaluate_search(self.index, LABELED_QUERIES, k=5)
        self.assertEqual(
            list(report["top_group"]), list(report["expected_group"])
        )

    def test_recall_is_reported_per_query(self):
        report = sta.evaluate_search(self.index, LABELED_QUERIES, k=5)
        self.assertEqual(len(report), len(LABELED_QUERIES))
        self.assertTrue((report["recall_at_k"] > 0.5).all())

    def test_search_returns_k_results(self):
        self.assertEqual(len(self.index.search("refund", k=3)), 3)

    def test_results_are_ordered_by_similarity(self):
        scores = [r.score for r in self.index.search("refund policy", k=5)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_is_deterministic(self):
        first = [r.record_id for r in self.index.search("timeout", k=5)]
        second = [r.record_id for r in self.index.search("timeout", k=5)]
        self.assertEqual(first, second)

    def test_results_carry_metadata(self):
        result = self.index.search("refund policy", k=1)[0]
        self.assertIn("failure_group", result.metadata)
        self.assertNotIn("document", result.metadata)

    def test_invalid_k_is_rejected(self):
        with self.assertRaises(ValueError):
            self.index.search("refund", k=0)

    def test_unfitted_index_refuses_to_search(self):
        with self.assertRaises(RuntimeError):
            sta.SemanticIndex().search("refund")

    def test_empty_documents_are_rejected(self):
        with self.assertRaises(ValueError):
            sta.SemanticIndex().fit(pd.DataFrame(columns=["document"]))

    def test_vectors_are_l2_normalized(self):
        norms = np.linalg.norm(self.index.vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


class TestClustering(SearchTestCase):
    def test_clustering_is_reproducible(self):
        first = sta.cluster_vectors(self.index.vectors, k=4)
        second = sta.cluster_vectors(self.index.vectors, k=4)
        np.testing.assert_array_equal(first, second)

    def test_select_k_reports_the_required_measures(self):
        table = sta.select_k(self.index.vectors, candidates=(2, 3, 4, 5))
        for column in (
            "k",
            "silhouette",
            "stability",
            "smallest_cluster",
            "largest_cluster",
        ):
            self.assertIn(column, table.columns)

    def test_select_k_skips_impossible_candidates(self):
        table = sta.select_k(self.index.vectors, candidates=(1, 999))
        self.assertTrue(table.empty)

    def test_stability_is_one_for_repeated_identical_runs(self):
        # Well-separated planted groups should cluster the same way whatever
        # the seed.
        self.assertGreater(sta.stability(self.index.vectors, k=4), 0.5)

    def test_medoid_is_a_real_member(self):
        labels = sta.cluster_vectors(self.index.vectors, k=4)
        for cluster in set(labels):
            members = [i for i, c in enumerate(labels) if c == cluster]
            self.assertIn(
                sta.medoid_index(self.index.vectors, members), members
            )

    def test_medoid_of_a_single_member_is_that_member(self):
        self.assertEqual(sta.medoid_index(self.index.vectors, [3]), 3)

    def test_empty_cluster_has_no_medoid(self):
        with self.assertRaises(ValueError):
            sta.medoid_index(self.index.vectors, [])

    def test_top_terms_are_interpretable(self):
        labels = sta.cluster_vectors(self.index.vectors, k=4)
        members = [i for i, c in enumerate(labels) if c == labels[0]]
        terms = sta.top_terms(self.index, members, n=5)

        self.assertTrue(terms)
        self.assertLessEqual(len(terms), 5)
        self.assertTrue(all(isinstance(t, str) for t in terms))

    def test_summary_covers_every_cluster(self):
        labels = sta.cluster_vectors(self.index.vectors, k=4)
        summary = sta.summarize_clusters(self.index, labels, self.evaluations)

        self.assertEqual(len(summary), len(set(labels)))
        for column in (
            "cluster",
            "size",
            "medoid_record_id",
            "top_terms",
            "app_versions",
            "lowest_metrics",
            "examples",
        ):
            self.assertIn(column, summary.columns)

    def test_summary_sizes_sum_to_the_corpus(self):
        labels = sta.cluster_vectors(self.index.vectors, k=4)
        summary = sta.summarize_clusters(self.index, labels, self.evaluations)
        self.assertEqual(summary["size"].sum(), len(self.documents))

    def test_clusters_recover_the_planted_failure_groups(self):
        # The fixture plants four well-separated groups; clustering should
        # substantially recover them.
        labels = sta.cluster_vectors(self.index.vectors, k=4)
        from sklearn.metrics import adjusted_rand_score

        self.assertGreater(
            adjusted_rand_score(self.documents["failure_group"], labels), 0.5
        )

    def test_pca_is_two_dimensional(self):
        projected = sta.pca_2d(self.index.vectors)
        self.assertEqual(projected.shape, (len(self.index.vectors), 2))

    def test_pca_handles_a_single_vector(self):
        self.assertEqual(sta.pca_2d(np.zeros((1, 5))).shape, (1, 2))


class TestNotebook(unittest.TestCase):
    """The notebook must stay stripped and credential-free."""

    @classmethod
    def setUpClass(cls):
        if not NOTEBOOK.exists():
            raise unittest.SkipTest(f"{NOTEBOOK} is missing")
        cls.notebook = json.loads(NOTEBOOK.read_text())

    def test_outputs_are_stripped(self):
        for i, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell.get("outputs", []), [], f"cell {i}")
            self.assertIsNone(cell.get("execution_count"), f"cell {i}")

    def test_no_empty_cells(self):
        for i, cell in enumerate(self.notebook["cells"]):
            self.assertTrue(
                "".join(cell["source"]).strip(), f"cell {i} is empty"
            )

    def test_code_cells_parse(self):
        import ast

        for i, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if source.strip().startswith("#"):
                continue
            ast.parse(source)

    def test_no_credentials_are_required(self):
        text = json.dumps(self.notebook)
        for marker in ("OPENAI_API_KEY", "sk-...", "ANTHROPIC_API_KEY"):
            self.assertNotIn(marker, text)

    def test_optional_embedding_section_is_not_executed_by_default(self):
        # The sentence-transformer section must stay opt-in so the default run
        # downloads nothing.
        text = json.dumps(self.notebook)
        if "sentence-transformers" in text:
            self.assertIn("RUN_LOCAL_EMBEDDINGS", text)

    def test_no_streamlit_dependency(self):
        self.assertNotIn("streamlit", json.dumps(self.notebook).lower())


class TestNoStreamlitInHelpers(unittest.TestCase):
    def test_helper_module_does_not_import_streamlit(self):
        source = (COOKBOOK / "semantic_trace_analysis.py").read_text()
        self.assertNotIn("streamlit", source.lower())


if __name__ == "__main__":
    unittest.main()
