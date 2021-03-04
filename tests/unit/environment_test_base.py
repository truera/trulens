import numpy as np
import os
import importlib
import trulens

from trulens.nn.backend import get_backend
from trulens.nn.models import get_model_wrapper

class EnvironmentTestBase(object):
    def setUp(self):
        self.input_size = 5
        self.output_size = 3

    def test_model_wrapper(self):
        for model in self.models:
            os.environ['TRULENS_BACKEND'] = self.incorrect_backend
            raised_error = False
            try:
                incorrect_model_wrapper = get_model_wrapper(model, backend=self.incorrect_backend)
            except:
                raised_error = True
            self.assertEqual(get_backend().backend, self.incorrect_backend)
            model_wrapper = get_model_wrapper(model)
            self.assertEqual(get_backend().backend, self.correct_backend)
            if not raised_error:
                self.assertNotEqual(type(model_wrapper), type(incorrect_model_wrapper))
    
    def test_backend(self):
        os.environ['TRULENS_BACKEND'] = self.incorrect_backend
        self.assertEqual(get_backend().backend, self.incorrect_backend)
        os.environ['TRULENS_BACKEND'] = self.correct_backend
        self.assertEqual(get_backend().backend, self.correct_backend)




