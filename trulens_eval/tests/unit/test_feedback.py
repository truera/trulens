"""
Tests for Feedback class. 
"""

from unittest import main
from unittest import TestCase


from trulens_eval import Provider, Feedback
from trulens_eval.util import jsonify
from trulens_eval.tru_basic_app import TruBasicApp

# Get the "globally importable" feedback implementations.
from tests.unit.feedbacks import custom_feedback_function, CustomProvider, CustomClass

class FeedbackConstructors(TestCase):

    def setUp(self):
        self.app = TruBasicApp(text_to_text = lambda t: f"returning {t}")
        _, self.record = self.app.call_with_record(input="hello")

    def test_feedback_functions(self):
        for imp, target in [
            (custom_feedback_function, 0.1),
            (CustomProvider.custom_provider_static_method, 0.2),
            (CustomProvider().custom_provider_method, 0.3),
            (CustomClass.custom_class_static_method, 0.4),
            (CustomClass().custom_class_method, 0.5)
        ]:

            with self.subTest(imp=imp,taget=target):
                f = Feedback(imp).on_default()

                # Run the feedback function.
                res = f.run(record=self.record, app=self.app)

                self.assertEqual(res.result, target)

                # Serialize and deserialize the feedback function.
                fs = jsonify(f)
                fds = Feedback(**fs)

                # Run it again.
                res = fds.run(record=self.record, app=self.app)

                self.assertEqual(res.result, target)


if __name__ == '__main__':
    main()
