import pandas as pd
from trulens.apps.app import TruApp
from trulens.core.feedback import Feedback
from trulens.core.feedback.selector import Selector
from trulens.core.otel.instrument import instrument

from tests.util.otel_test_case import OtelTestCase


class TestOtelRecording(OtelTestCase):
    def test_multi_records_and_recordings(self) -> None:
        # Create feedbacks.
        def baby_grader(baby: str) -> float:
            return 1.0

        def char_counter(text: str) -> float:
            return float(len(text))

        f_baby_grader = Feedback(baby_grader, name="Baby Grader").on({
            "baby": Selector.select_record_input()
        })
        f_char_counter = Feedback(char_counter, name="Char Counter").on({
            "text": Selector.select_record_output()
        })

        # Create app.
        class SimpleApp:
            @instrument()
            def greet(self, name: str) -> str:
                return f"Hello, {name}!"

        app = SimpleApp()
        tru_app = TruApp(
            app,
            app_name="SimpleApp",
            app_version="v1",
            feedbacks=[f_baby_grader, f_char_counter],
        )
        # Invoke and record.
        with tru_app as recording1:
            app.greet("Kojikun")
            app.greet("Nolan")
        with tru_app as recording2:
            app.greet("Sachiboy")
        # Create expected all recordings.
        expected_all_recordings = pd.DataFrame(
            {
                "Baby Grader": [1.0, 1.0, 1.0],
                "Char Counter": [
                    float(len(f"Hello, {name}!"))
                    for name in ["Kojikun", "Nolan", "Sachiboy"]
                ],
            },
            index=pd.Series(
                [
                    recording1[0].record_id,
                    recording1[1].record_id,
                    recording2[0].record_id,
                ],
                name="record_id",
            ),
        )
        # Verify `recording1`.
        with self.assertRaises(RuntimeError):
            recording1.get()
        self.assertEqual(2, len(recording1))
        self.assertEqual(2, len(recording1.records))
        import trulens.core.otel.recording as recording_module

        print("DEBUG recording module file", recording_module.__file__)
        print("DEBUG type recording1[-1]", type(recording1[-1]))
        res = recording1[-1].retrieve_feedback_results()
        print("DEBUG recording1[-1] res", res)
        pd.testing.assert_frame_equal(expected_all_recordings.iloc[1:2], res)
        res = recording1.retrieve_feedback_results()
        pd.testing.assert_frame_equal(expected_all_recordings.iloc[:2], res)
        # Verify `recording2`.
        self.assertEqual(1, len(recording2))
        self.assertEqual(1, len(recording2.records))
        res = recording2.get().retrieve_feedback_results()
        pd.testing.assert_frame_equal(expected_all_recordings.iloc[2:], res)
        # Verify entire app recording.
        res = tru_app.retrieve_feedback_results()
        pd.testing.assert_frame_equal(expected_all_recordings, res)
