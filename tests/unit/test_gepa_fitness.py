"""Unit tests for trulens-apps-gepa (TruGEPA and run_evolution)."""

from unittest import TestCase
from unittest.mock import MagicMock
from unittest.mock import patch

from trulens.apps.gepa.fitness import TruGEPA
from trulens.apps.gepa.fitness import run_evolution


def _fixed_score(prompt: str) -> float:
    return 0.7


def _tuple_score(prompt: str):
    return (0.6, {"reason": "ok"})


def _length_score(prompt: str) -> float:
    return min(len(prompt) / 100.0, 1.0)


def _mutate(prompt: str) -> str:
    return prompt + " (improved)"


class TestTruGEPA(TestCase):
    def test_basic_call_returns_float(self):
        fitness = TruGEPA(_fixed_score, optimize_key="prompt")
        score = fitness("some prompt")
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 0.7)

    def test_tuple_result_extracts_first_element(self):
        fitness = TruGEPA(_tuple_score, optimize_key="prompt")
        score = fitness("some prompt")
        self.assertAlmostEqual(score, 0.6)

    def test_optimize_key_forwarded_correctly(self):
        received = {}

        def capture(**kwargs):
            received.update(kwargs)
            return 0.5

        fitness = TruGEPA(capture, optimize_key="question")
        fitness("hello")
        self.assertIn("question", received)
        self.assertEqual(received["question"], "hello")

    def test_feedback_args_forwarded(self):
        received = {}

        def capture(**kwargs):
            received.update(kwargs)
            return 0.5

        fitness = TruGEPA(
            capture,
            optimize_key="prompt",
            feedback_args={"context": "some background", "extra": 42},
        )
        fitness("hello")
        self.assertEqual(received.get("context"), "some background")
        self.assertEqual(received.get("extra"), 42)

    def test_no_feedback_args_by_default(self):
        received = {}

        def capture(**kwargs):
            received.update(kwargs)
            return 0.5

        fitness = TruGEPA(capture, optimize_key="prompt")
        fitness("hello")
        self.assertEqual(set(received.keys()), {"prompt"})

    def test_extra_kwargs_forwarded(self):
        received = {}

        def capture(**kwargs):
            received.update(kwargs)
            return 0.5

        fitness = TruGEPA(capture, optimize_key="prompt")
        fitness("hello", temperature=0.0)
        self.assertEqual(received.get("temperature"), 0.0)

    # --- logging behaviour ---

    def test_no_logging_when_neither_app_name_nor_version_supplied(self):
        fitness = TruGEPA(_fixed_score)
        self.assertIsNone(fitness._recorder)
        score = fitness("prompt")
        self.assertAlmostEqual(score, 0.7)

    def test_raises_when_only_app_name_supplied(self):
        with self.assertRaises(ValueError) as ctx:
            TruGEPA(_fixed_score, app_name="optimizer")
        self.assertIn("app_name", str(ctx.exception))
        self.assertIn("app_version", str(ctx.exception))

    def test_raises_when_only_app_version_supplied(self):
        with self.assertRaises(ValueError) as ctx:
            TruGEPA(_fixed_score, app_version="v1")
        self.assertIn("app_name", str(ctx.exception))
        self.assertIn("app_version", str(ctx.exception))

    def test_recorder_created_eagerly_with_both_params(self):
        with patch("trulens.apps.virtual.TruVirtual") as MockTV:
            fitness = TruGEPA(
                _fixed_score, app_name="my_optimizer", app_version="v2"
            )
        MockTV.assert_called_once_with(
            app_name="my_optimizer", app_version="v2"
        )
        self.assertIsNotNone(fitness._recorder)

    def test_add_record_called_on_every_evaluation_when_logging_enabled(self):
        with patch("trulens.apps.virtual.TruVirtual") as MockTV, patch(
            "trulens.apps.virtual.VirtualRecord"
        ), patch("trulens.core.Select"):
            mock_recorder = MockTV.return_value
            fitness = TruGEPA(_fixed_score, app_name="opt", app_version="v1")
            fitness("a")
            fitness("b")

        self.assertEqual(mock_recorder.add_record.call_count, 2)

    def test_no_add_record_when_logging_disabled(self):
        with patch("trulens.apps.virtual.TruVirtual") as MockTV:
            fitness = TruGEPA(_fixed_score)
            fitness("prompt")
        MockTV.assert_not_called()

    def test_logging_failure_does_not_raise(self):
        with patch("trulens.apps.virtual.TruVirtual"), patch(
            "trulens.apps.virtual.VirtualRecord"
        ), patch("trulens.core.Select") as MockSelect:
            MockSelect.RecordCalls.fitness_fn.evaluate = MagicMock()
            fitness = TruGEPA(_fixed_score, app_name="opt", app_version="v1")
            fitness._recorder.add_record.side_effect = RuntimeError("db error")
            score = fitness("prompt")
        self.assertAlmostEqual(score, 0.7)


class TestRunEvolution(TestCase):
    def _make_fitness(self, fn):
        """Wrap a score fn in TruGEPA with logging disabled."""
        return TruGEPA(fn)

    def test_returns_correct_structure(self):
        fitness = self._make_fitness(_fixed_score)
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
        fitness = self._make_fitness(_length_score)
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
        fitness = self._make_fitness(_length_score)
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
        fitness = self._make_fitness(_fixed_score)
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
        fitness = self._make_fitness(_length_score)
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
        """run_evolution should accept any plain callable, not just TruGEPA."""
        best, score, _ = run_evolution(
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
