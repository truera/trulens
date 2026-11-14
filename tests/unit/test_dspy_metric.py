"""Unit tests for trulens-apps-dspy (TruDSPyMetric)."""

import json
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from typing import ClassVar
from unittest import TestCase
import warnings

import numpy as np
import pytest
from trulens.apps.dspy import TruDSPyMetric
from trulens.core import Feedback
from trulens.core import Metric
from trulens.core.utils.constants import UNPARSABLE_SCORE
from trulens.feedback import llm_provider

QA_FIELDS = {"question": "example.question", "answer": "prediction.answer"}

ANSWERS = {
    "What is the capital of France?": "Paris",
    "What is two plus two?": "Four",
    "What color is the sky?": "Green",
}


def _judge(question: str, answer: str):
    score = 0.9 if answer in ("Paris", "Four") else 0.1
    return score, {"reason": f"{answer!r} judged against {question!r}."}


def _overlap(question: str, answer: str) -> float:
    return 0.5


def _returning(value):
    def implementation(question: str, answer: str):
        return value

    return implementation


def _ns(**fields):
    return SimpleNamespace(**fields)


class _MockEndpoint:
    def run_in_pace(self, func, *args, **kwargs):
        return func(*args, **kwargs)


class _CannedJudge(llm_provider.LLMProvider):
    """A real LLM provider whose chat completion returns a fixed reply."""

    model_config: ClassVar[dict] = {"extra": "allow"}

    def __init__(self, reply: str):
        super().__init__(endpoint=None, model_engine="mock-model")
        object.__setattr__(self, "endpoint", _MockEndpoint())
        object.__setattr__(self, "_reply", reply)
        object.__setattr__(self, "prompts", [])

    def _is_reasoning_model(self) -> bool:
        return False

    def _create_chat_completion(
        self, prompt=None, messages=None, response_format=None, **kwargs
    ):
        self.prompts.append(messages)
        return self._reply


_RELEVANCE_REPLY = json.dumps({
    "criteria": "The answer must address the question.",
    "supporting_evidence": "The answer names Paris, the capital asked for.",
    "score": 2,
})


def _relevance_metric(judge: _CannedJudge) -> TruDSPyMetric:
    return TruDSPyMetric(
        Metric(
            implementation=judge.relevance_with_cot_reasons,
            name="Answer Relevance",
        ),
        fields={"prompt": "example.question", "response": "prediction.answer"},
    )


class TestFieldMapping(TestCase):
    def test_paths_read_example_and_prediction_fields(self):
        received = {}

        def capture(question: str, answer: str) -> float:
            received.update(question=question, answer=answer)
            return 0.5

        metric = TruDSPyMetric(capture, fields=QA_FIELDS)
        metric(_ns(question="q?"), _ns(answer="a."))
        self.assertEqual(received, {"question": "q?", "answer": "a."})

    def test_nested_path(self):
        received = {}

        def capture(question: str, answer: str) -> float:
            received["answer"] = answer
            return 0.5

        metric = TruDSPyMetric(
            capture,
            fields={
                "question": "example.question",
                "answer": "prediction.output.text",
            },
        )
        prediction = _ns(output=_ns(text="nested"))
        metric(_ns(question="q?"), prediction)
        self.assertEqual(received["answer"], "nested")

    def test_callable_source_receives_example_and_prediction(self):
        received = {}

        def capture(source: str, statement: str) -> float:
            received["source"] = source
            return 0.5

        metric = TruDSPyMetric(
            capture,
            fields={
                "source": lambda example, prediction: "\n".join(
                    prediction.passages
                ),
                "statement": "prediction.answer",
            },
        )
        metric(_ns(), _ns(passages=["p1", "p2"], answer="a"))
        self.assertEqual(received["source"], "p1\np2")

    def test_missing_field_names_argument_and_path(self):
        metric = TruDSPyMetric(_overlap, fields=QA_FIELDS)
        with self.assertRaises(ValueError) as ctx:
            metric(_ns(question="q?"), _ns(text="a."))
        message = str(ctx.exception)
        self.assertIn("'answer'", message)
        self.assertIn("'prediction.answer'", message)

    def test_bound_arguments_are_passed(self):
        received = {}

        def capture(question: str, answer: str, style: str) -> float:
            received["style"] = style
            return 0.5

        metric = TruDSPyMetric(
            Metric(implementation=capture).with_arguments(style="strict"),
            fields=QA_FIELDS,
        )
        metric(_ns(question="q?"), _ns(answer="a."))
        self.assertEqual(received["style"], "strict")

    def test_metric_setting_fills_required_argument(self):
        received = {}

        def capture(question: str, answer: str, criteria: str) -> float:
            received["criteria"] = criteria
            return 0.5

        metric = TruDSPyMetric(
            Metric(implementation=capture, criteria="Be brief."),
            fields=QA_FIELDS,
        )
        metric(_ns(question="q?"), _ns(answer="a."))
        self.assertEqual(received["criteria"], "Be brief.")

    def test_optional_argument_may_be_unmapped(self):
        def implementation(question: str, answer: str, extra: str = "x"):
            return 0.5

        metric = TruDSPyMetric(implementation, fields=QA_FIELDS)
        self.assertEqual(metric(_ns(question="q"), _ns(answer="a")), 0.5)


class TestValidation(TestCase):
    def test_unknown_argument_raises(self):
        with self.assertRaises(ValueError) as ctx:
            TruDSPyMetric(
                _overlap, fields={**QA_FIELDS, "context": "example.context"}
            )
        self.assertIn("context", str(ctx.exception))

    def test_unknown_argument_raises_with_var_kwargs(self):
        # Metric.__call__ forwards only named parameters, so the extra field
        # would never reach the implementation.
        def implementation(question: str, answer: str, **kwargs) -> float:
            return 0.5

        with self.assertRaises(ValueError) as ctx:
            TruDSPyMetric(
                implementation,
                fields={**QA_FIELDS, "context": "example.context"},
            )
        self.assertIn("context", str(ctx.exception))

    def test_bound_argument_in_fields_raises(self):
        def implementation(question: str, answer: str, style: str) -> float:
            return 0.5

        metric = Metric(implementation=implementation).with_arguments(
            style="strict"
        )
        with self.assertRaises(ValueError) as ctx:
            TruDSPyMetric(
                metric, fields={**QA_FIELDS, "style": "example.style"}
            )
        self.assertIn("style", str(ctx.exception))
        self.assertIn("with_arguments", str(ctx.exception))

    def test_field_replaced_by_metric_setting_raises(self):
        # Metric.__call__ would pass the metric's criteria over the mapped one.
        def implementation(question: str, answer: str, criteria=None):
            return 0.5

        with self.assertRaises(ValueError) as ctx:
            TruDSPyMetric(
                Metric(implementation=implementation, criteria="Be brief."),
                fields={**QA_FIELDS, "criteria": "example.rubric"},
            )
        self.assertIn("criteria", str(ctx.exception))
        # Unset, the setting leaves the field to the mapping.
        TruDSPyMetric(
            implementation, fields={**QA_FIELDS, "criteria": "example.rubric"}
        )

    def test_unmapped_required_argument_raises(self):
        with self.assertRaises(ValueError) as ctx:
            TruDSPyMetric(_overlap, fields={"question": "example.question"})
        self.assertIn("answer", str(ctx.exception))

    def test_nothing_is_inferred_from_argument_names(self):
        with self.assertRaises(ValueError):
            TruDSPyMetric(_overlap, fields={})

    def test_malformed_paths_raise(self):
        for path in (
            "answer",
            "example",
            "pred.answer",
            "prediction.",
            "prediction..answer",
            "example.1st",
            "example.question.",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError) as ctx:
                    TruDSPyMetric(
                        _overlap,
                        fields={"question": path, "answer": "prediction.a"},
                    )
                self.assertIn(repr(path), str(ctx.exception))

    def test_invalid_source_type_raises(self):
        with self.assertRaises(TypeError):
            TruDSPyMetric(_overlap, fields={"question": 1, "answer": "x.y"})

    def test_fields_must_be_a_mapping(self):
        with self.assertRaises(TypeError):
            TruDSPyMetric(_overlap, fields=["question", "answer"])

    def test_metric_must_be_callable(self):
        with self.assertRaises(TypeError):
            TruDSPyMetric("relevance", fields=QA_FIELDS)


class TestMetricWrapping(TestCase):
    def test_plain_callable_is_wrapped_in_metric(self):
        metric = TruDSPyMetric(_overlap, fields=QA_FIELDS)
        self.assertIsInstance(metric._metric, Metric)
        self.assertEqual(metric.name, "_overlap")
        self.assertEqual(metric.__name__, "_overlap")

    def test_metric_name_is_exposed(self):
        metric = TruDSPyMetric(
            Metric(implementation=_overlap, name="Overlap"), fields=QA_FIELDS
        )
        self.assertEqual(metric.name, "Overlap")
        self.assertEqual(metric.__name__, "Overlap")

    def test_feedback_subclass_is_accepted(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = Feedback(_overlap, name="Legacy")
        metric = TruDSPyMetric(legacy, fields=QA_FIELDS)
        self.assertEqual(metric.name, "Legacy")
        self.assertEqual(metric(_ns(question="q"), _ns(answer="a")), 0.5)


class TestScoreNormalization(TestCase):
    def _score(self, value, **metric_kwargs):
        metric = TruDSPyMetric(
            Metric(implementation=_returning(value), **metric_kwargs),
            fields=QA_FIELDS,
        )
        return metric(_ns(question="q"), _ns(answer="a"))

    def test_numbers_become_floats(self):
        for value, expected in (
            (0.25, 0.25),
            (1, 1.0),
            (True, 1.0),
            (False, 0.0),
            (np.float32(0.5), 0.5),
            (np.int64(1), 1.0),
            (np.bool_(True), 1.0),
            (np.array(0.75), 0.75),
        ):
            with self.subTest(value=value):
                score = self._score(value)
                self.assertIs(type(score), float)
                self.assertAlmostEqual(score, expected)

    def test_tuple_with_metadata_returns_score(self):
        self.assertAlmostEqual(self._score((0.6, {"reason": "ok"})), 0.6)

    def test_sequence_uses_metric_agg(self):
        self.assertAlmostEqual(self._score([0.2, 0.4]), 0.3)
        self.assertAlmostEqual(self._score(np.array([0.2, 0.4])), 0.3)
        self.assertAlmostEqual(self._score(([0.2, 0.4], {"reason": "x"})), 0.3)
        self.assertAlmostEqual(self._score([0.2, 0.4], agg=np.min), 0.2)

    def test_sequence_skips_unparsable_sentinel(self):
        self.assertAlmostEqual(self._score([1.0, UNPARSABLE_SCORE]), 1.0)

    def test_failed_evaluations_raise(self):
        for value in (
            UNPARSABLE_SCORE,
            [UNPARSABLE_SCORE, UNPARSABLE_SCORE],
            float("nan"),
            float("inf"),
            (float("nan"), {"reason": "x"}),
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as ctx:
                    self._score(value)
                self.assertIn("no usable score", str(ctx.exception))

    def test_results_without_one_score_raise(self):
        for value in (
            {"a": 0.5, "b": 1.0},
            "0.5",
            None,
            [],
            [[0.5], [1.0]],
            [0.5, "high"],
            (0.5, 0.7, 0.9, "x"),
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as ctx:
                    self._score(value)
                self.assertIn(
                    "cannot be reduced to one score", str(ctx.exception)
                )


class TestThreshold(TestCase):
    def setUp(self):
        self.example = _ns(question="q")

    def test_float_without_trace(self):
        metric = TruDSPyMetric(_overlap, fields=QA_FIELDS, threshold=0.7)
        score = metric(self.example, _ns(answer="a"))
        self.assertIs(type(score), float)
        self.assertEqual(score, 0.5)
        self.assertEqual(metric(self.example, _ns(answer="a"), None), 0.5)

    def test_bool_with_trace(self):
        prediction = _ns(answer="a")
        below = TruDSPyMetric(_overlap, fields=QA_FIELDS, threshold=0.7)
        at = TruDSPyMetric(_overlap, fields=QA_FIELDS, threshold=0.5)
        self.assertIs(below(self.example, prediction, []), False)
        self.assertIs(at(self.example, prediction, []), True)

    def test_float_with_trace_and_no_threshold(self):
        metric = TruDSPyMetric(_overlap, fields=QA_FIELDS)
        self.assertEqual(metric(self.example, _ns(answer="a"), []), 0.5)


class TestFeedbackWithoutReasons(TestCase):
    def test_float_when_result_has_no_reasons(self):
        for value in (0.5, (0.5, {}), (0.5, {"score_detail": 1})):
            with self.subTest(value=value):
                metric = TruDSPyMetric(_returning(value), fields=QA_FIELDS)
                score = metric(_ns(question="q"), _ns(answer="a"), [], "qa", [])
                self.assertEqual(score, 0.5)

    def test_threshold_does_not_apply_to_feedback_calls(self):
        metric = TruDSPyMetric(_overlap, fields=QA_FIELDS, threshold=0.7)
        score = metric(_ns(question="q"), _ns(answer="a"), [], "qa", [])
        self.assertEqual(score, 0.5)


class TestStubProvider(TestCase):
    def test_real_provider_method_scores_mapped_fields(self):
        judge = _CannedJudge(_RELEVANCE_REPLY)
        metric = _relevance_metric(judge)
        score = metric(
            _ns(question="What is the capital of France?"),
            _ns(answer="Paris"),
        )
        self.assertAlmostEqual(score, 2 / 3)
        user_prompt = judge.prompts[-1][-1]["content"]
        self.assertIn("What is the capital of France?", user_prompt)
        self.assertIn("Paris", user_prompt)

    def test_unparsable_judge_reply_raises(self):
        judge = _CannedJudge(
            '{"criteria": "c", "supporting_evidence": "e", "score": "none"}'
        )
        metric = _relevance_metric(judge)
        with self.assertRaises(ValueError) as ctx:
            metric(_ns(question="q"), _ns(answer="a"))
        self.assertIn("no usable score", str(ctx.exception))


class TestImport(TestCase):
    def test_import_and_scoring_do_not_need_dspy(self):
        script = textwrap.dedent(
            """
            import sys

            # Any attempt to import dspy now raises ImportError.
            sys.modules["dspy"] = None

            from types import SimpleNamespace

            from trulens.apps.dspy import TruDSPyMetric

            def overlap(question, answer):
                return 0.5, {}

            metric = TruDSPyMetric(
                overlap,
                fields={
                    "question": "example.question",
                    "answer": "prediction.answer",
                },
                threshold=0.4,
            )
            example = SimpleNamespace(question="q")
            prediction = SimpleNamespace(answer="a")
            assert metric(example, prediction) == 0.5
            assert metric(example, prediction, []) is True
            assert metric(example, prediction, [], "qa", []) == 0.5
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class TestWithDSPy(TestCase):
    """Runs against real DSPy objects and optimizers; skipped without DSPy."""

    def setUp(self):
        self.dspy = pytest.importorskip("dspy")
        dspy = self.dspy
        self.examples = [
            dspy.Example(question=q, answer=a).with_inputs("question")
            for q, a in ANSWERS.items()
        ]
        self.metric = TruDSPyMetric(_judge, fields=QA_FIELDS)

    def _lm(self):
        from dspy.utils import DummyLM

        return DummyLM({q: {"answer": a} for q, a in ANSWERS.items()})

    def _program(self):
        dspy = self.dspy

        class QA(dspy.Module):
            def __init__(self):
                super().__init__()
                self.qa = dspy.Predict("question -> answer")

            def forward(self, question):
                return self.qa(question=question)

        return QA()

    def test_reads_real_example_and_prediction(self):
        score = self.metric(
            self.examples[0], self.dspy.Prediction(answer="Paris")
        )
        self.assertEqual(score, 0.9)

    def test_fields_named_like_example_methods(self):
        received = {}

        def capture(question, answer) -> float:
            received.update(question=question, answer=answer)
            return 0.5

        metric = TruDSPyMetric(
            capture,
            fields={"question": "example.labels", "answer": "prediction.items"},
        )
        metric(
            self.dspy.Example(question="q", labels="gold"),
            self.dspy.Prediction(items=["a", "b"]),
        )
        self.assertEqual(received, {"question": "gold", "answer": ["a", "b"]})

    def test_feedback_call_reports_unusable_score(self):
        # GEPA does not catch errors from its feedback call, so a failed judge
        # becomes feedback there while plain scoring still raises.
        metric = TruDSPyMetric(
            Metric(
                implementation=_returning((UNPARSABLE_SCORE, {"reason": "x"})),
                name="Groundedness",
            ),
            fields=QA_FIELDS,
        )
        example = self.examples[0]
        prediction = self.dspy.Prediction(answer="a")
        with self.assertLogs("trulens.apps.dspy.metric", "WARNING"):
            feedback = metric(example, prediction, [], "qa", [])
        self.assertIsInstance(feedback, self.dspy.Prediction)
        self.assertEqual(feedback.score, 0.0)
        self.assertEqual(
            feedback.feedback, "Groundedness could not score this example."
        )
        with self.assertRaises(ValueError):
            metric(example, prediction)

    def test_feedback_prediction_for_gepa(self):
        feedback = self.metric(
            self.examples[2], self.dspy.Prediction(answer="Green"), [], "qa", []
        )
        self.assertIsInstance(feedback, self.dspy.Prediction)
        self.assertEqual(feedback.score, 0.1)
        self.assertEqual(float(feedback), 0.1)
        # GEPA checks `hasattr(o, "feedback")` and reads `o["feedback"]`.
        self.assertTrue(hasattr(feedback, "feedback"))
        self.assertEqual(
            feedback["feedback"],
            "_judge score: 0.10\n"
            "'Green' judged against 'What color is the sky?'.",
        )

    def test_feedback_renders_structured_reasons(self):
        reasons = [
            {"criteria": "Sky is blue.", "supporting_evidence": "Yes."},
            {"criteria": "Grass is red.", "supporting_evidence": "No."},
        ]
        metric = TruDSPyMetric(
            Metric(
                implementation=_returning((0.5, {"reasons": reasons})),
                name="Groundedness",
            ),
            fields=QA_FIELDS,
        )
        feedback = metric(
            self.examples[0], self.dspy.Prediction(answer="a"), None, "qa", None
        )
        self.assertEqual(
            feedback.feedback,
            "Groundedness score: 0.50\n"
            "criteria: Sky is blue.\nsupporting_evidence: Yes.\n\n"
            "criteria: Grass is red.\nsupporting_evidence: No.",
        )

    def test_stub_provider_reasons_reach_feedback(self):
        metric = _relevance_metric(_CannedJudge(_RELEVANCE_REPLY))
        feedback = metric(
            self.examples[0], self.dspy.Prediction(answer="Paris"), [], "qa", []
        )
        self.assertAlmostEqual(feedback.score, 2 / 3)
        self.assertTrue(feedback.feedback.startswith("Answer Relevance score"))
        self.assertIn(
            "Supporting Evidence: The answer names Paris", feedback.feedback
        )

    def test_evaluate_scores_static_program(self):
        dspy = self.dspy

        class Static(dspy.Module):
            def forward(self, question):
                return dspy.Prediction(answer=ANSWERS[question])

        result = dspy.Evaluate(
            devset=self.examples,
            metric=self.metric,
            num_threads=1,
            display_progress=False,
        )(Static())
        scores = [score for *_, score in result.results]
        self.assertEqual(scores, [0.9, 0.9, 0.1])
        self.assertAlmostEqual(result.score, round(100 * 1.9 / 3, 2))

    def test_bootstrap_few_shot_keeps_demos_above_threshold(self):
        dspy = self.dspy

        def demos(metric):
            optimizer = dspy.BootstrapFewShot(
                metric=metric, max_bootstrapped_demos=3, max_labeled_demos=0
            )
            with dspy.context(lm=self._lm()):
                compiled = optimizer.compile(
                    self._program(), trainset=self.examples
                )
            return [demo.question for demo in compiled.qa.demos]

        gated = TruDSPyMetric(_judge, fields=QA_FIELDS, threshold=0.5)
        self.assertEqual(
            demos(gated),
            ["What is the capital of France?", "What is two plus two?"],
        )
        # Without a threshold, any nonzero float counts as a pass.
        self.assertEqual(len(demos(self.metric)), 3)

    def test_gepa_accepts_metric_signature(self):
        dspy = self.dspy

        def proposer(candidate, reflective_dataset, components_to_update):
            return dict(candidate)

        dspy.GEPA(
            metric=self.metric, instruction_proposer=proposer, auto="light"
        )
        with self.assertRaises(TypeError):
            dspy.GEPA(
                metric=lambda example, prediction: 0.0,
                instruction_proposer=proposer,
                auto="light",
            )

    def test_gepa_reflection_receives_trulens_reasons(self):
        dspy = self.dspy
        datasets = []

        def proposer(candidate, reflective_dataset, components_to_update):
            datasets.append(reflective_dataset)
            return {
                name: candidate[name] + " Answer correctly."
                for name in components_to_update
            }

        optimizer = dspy.GEPA(
            metric=self.metric,
            instruction_proposer=proposer,
            max_metric_calls=12,
            reflection_minibatch_size=3,
            num_threads=1,
        )
        with dspy.context(lm=self._lm()):
            optimizer.compile(
                self._program(), trainset=self.examples, valset=self.examples
            )

        self.assertTrue(datasets)
        feedback = {
            item["Inputs"]["question"]: item["Feedback"]
            for item in datasets[0]["qa"]
        }
        self.assertEqual(
            feedback["What color is the sky?"],
            "_judge score: 0.10\n"
            "'Green' judged against 'What color is the sky?'.",
        )
