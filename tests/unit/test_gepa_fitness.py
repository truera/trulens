"""Unit tests for trulens-apps-gepa (TruLensFitness and run_evolution)."""

from unittest import TestCase
from unittest.mock import MagicMock
from unittest.mock import patch

from trulens.apps.gepa.fitness import TruLensFitness
from trulens.apps.gepa.fitness import run_evolution


def _fixed_score(prompt: str) -> float:
    return 0.7


def _tuple_score(prompt: str):
    return (0.6, {"reason": "ok"})


def _length_score(prompt: str) -> float:
    return min(len(prompt) / 100.0, 1.0)


def _mutate(prompt: str) -> str:
    return prompt + " (improved)"


class TestTruLensFitness(TestCase):
    def test_basic_call_returns_float(self):
        fitness = TruLensFitness(_fixed_score, input_key="prompt")
        score = fitness("some prompt")
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 0.7)

    def test_tuple_result_extracts_first_element(self):
        fitness = TruLensFitness(_tuple_score, input_key="prompt")
        score = fitness("some prompt")
        self.assertAlmostEqual(score, 0.6)

    def test_input_key_forwarded_correctly(self):
        received = {}

        def capture(**kwargs):
            received.update(kwargs)
            return 0.5

        fitness = TruLensFitness(capture, input_key="question")
        fitness("hello")
        self.assertIn("question", received)
        self.assertEqual(received["question"], "hello")

    def test_context_forwarded_when_set(self):
        received = {}

        def capture(**kwargs):
            received.update(kwargs)
            return 0.5

        fitness = TruLensFitness(
            capture,
            input_key="prompt",
            context_key="context",
            context="some background",
        )
        fitness("hello")
        self.assertEqual(received.get("context"), "some background")

    def test_context_not_forwarded_when_absent(self):
        received = {}

        def capture(**kwargs):
            received.update(kwargs)
            return 0.5

        fitness = TruLensFitness(capture, input_key="prompt")
        fitness("hello")
        self.assertNotIn("context", received)

    def test_extra_kwargs_forwarded(self):
        received = {}

        def capture(**kwargs):
            received.update(kwargs)
            return 0.5

        fitness = TruLensFitness(capture, input_key="prompt")
        fitness("hello", temperature=0.0)
        self.assertEqual(received.get("temperature"), 0.0)

    def test_no_recorder_logs_nothing(self):
        fitness = TruLensFitness(_fixed_score)
        score = fitness("prompt")
        self.assertAlmostEqual(score, 0.7)

    def test_recorder_add_record_called(self):
        mock_recorder = MagicMock()
        fitness = TruLensFitness(_fixed_score, recorder=mock_recorder)

        with patch("trulens.apps.virtual.VirtualRecord") as MockVR:
            with patch("trulens.core.Select"):
                MockVR.return_value = MagicMock()
                fitness("test prompt")

        mock_recorder.add_record.assert_called_once()

    def test_recorder_failure_does_not_raise(self):
        mock_recorder = MagicMock()
        mock_recorder.add_record.side_effect = RuntimeError("db error")
        fitness = TruLensFitness(_fixed_score, recorder=mock_recorder)

        with patch("trulens.apps.virtual.VirtualRecord") as MockVR:
            with patch("trulens.core.Select"):
                MockVR.return_value = MagicMock()
                score = fitness("prompt")

        self.assertAlmostEqual(score, 0.7)


class TestRunEvolution(TestCase):
    def test_returns_correct_structure(self):
        fitness = TruLensFitness(_fixed_score)
        best_prompt, best_score, history = run_evolution(
            base_prompt="base",
            fitness_fn=fitness,
            mutate_fn=_mutate,
            n_generations=3,
            population_size=3,
            top_k=1,
            seed=42,
        )
        self.assertIsInstance(best_prompt, str)
        self.assertIsInstance(best_score, float)
        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 3)
        for entry in history:
            self.assertEqual(len(entry), 2)
            self.assertIsInstance(entry[0], str)
            self.assertIsInstance(entry[1], float)

    def test_best_score_is_max_of_history(self):
        fitness = TruLensFitness(_length_score)
        _, best_score, history = run_evolution(
            base_prompt="hi",
            fitness_fn=fitness,
            mutate_fn=_mutate,
            n_generations=5,
            population_size=3,
            top_k=1,
            seed=0,
        )
        self.assertAlmostEqual(best_score, max(s for _, s in history))

    def test_seed_produces_deterministic_results(self):
        fitness = TruLensFitness(_length_score)
        kwargs = dict(
            base_prompt="start",
            fitness_fn=fitness,
            mutate_fn=_mutate,
            n_generations=4,
            population_size=4,
            top_k=2,
            seed=123,
        )
        _, score1, history1 = run_evolution(**kwargs)
        _, score2, history2 = run_evolution(**kwargs)
        self.assertAlmostEqual(score1, score2)
        self.assertEqual(history1, history2)

    def test_top_k_must_be_less_than_population_size(self):
        fitness = TruLensFitness(_fixed_score)
        with self.assertRaises(ValueError):
            run_evolution(
                base_prompt="x",
                fitness_fn=fitness,
                mutate_fn=_mutate,
                population_size=3,
                top_k=3,
            )

    def test_evolution_improves_with_length_fitness(self):
        """Longer prompts score higher; mutating by appending should trend up."""
        fitness = TruLensFitness(_length_score)
        _, _, history = run_evolution(
            base_prompt="short",
            fitness_fn=fitness,
            mutate_fn=_mutate,
            n_generations=6,
            population_size=4,
            top_k=2,
            seed=7,
        )
        self.assertGreaterEqual(history[-1][1], history[0][1])

    def test_plain_callable_works_as_fitness(self):
        """run_evolution should accept any plain callable, not just TruLensFitness."""
        best, score, history = run_evolution(
            base_prompt="prompt",
            fitness_fn=lambda p: 0.5,
            mutate_fn=lambda p: p + "!",
            n_generations=2,
            population_size=2,
            top_k=1,
            seed=1,
        )
        self.assertIsInstance(best, str)
        self.assertAlmostEqual(score, 0.5)
