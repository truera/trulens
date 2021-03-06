import numpy as np
import os
import importlib
import trulens

from trulens.nn.backend import get_backend
from trulens.nn.models import get_model_wrapper


class EnvironmentTestBase(object):
    _environment_backends = ['pytorch', 'tensorflow', 'keras', 'tf.keras']

    def setUp(self):
        self.input_size = 5
        self.output_size = 3
        self.model_lin_weights = np.random.normal(
            scale=2. / (self.input_size + self.output_size),
            size=(self.input_size, self.output_size))
        self.model_lin_bias = np.random.uniform(-0.5, 0.5, (self.output_size,))

    def test_model_wrapper(self):
        for incorrect_backend in EnvironmentTestBase._environment_backends:
            if self.correct_backend == incorrect_backend:
                continue
            for i in range(len(self.models)):
                model = self.models[i]
                kwargs = self.models_wrapper_kwargs[i]
                os.environ['TRULENS_BACKEND'] = incorrect_backend
                raised_error = False
                try:
                    forced_backend_kwargs = kwargs.copy()
                    forced_backend_kwargs['backend'] = incorrect_backend
                    incorrect_model_wrapper = get_model_wrapper(
                        model, **forced_backend_kwargs)
                except:
                    raised_error = True

                # A None backend is a valid outcome for incorrect backend if imports fail
                if get_backend() is not None:
                    self.assertEqual(get_backend().backend, incorrect_backend)

                model_wrapper = get_model_wrapper(model, **kwargs)
                self.assertEqual(get_backend().backend, self.correct_backend)
                if not raised_error:
                    self.assertNotEqual(
                        type(model_wrapper), type(incorrect_model_wrapper))
                self.assertIsInstance(model_wrapper, self.model_wrapper_type)

    def test_model_wrapper_params(self):
        for i in range(len(self.models)):
            model = self.models[i]
            kwargs = self.models_wrapper_kwargs[i]
            for kwarg_key in kwargs:
                missing_kwarg_list = kwargs.copy()
                del missing_kwarg_list[kwarg_key]
                with self.assertRaises(ValueError):
                    get_model_wrapper(model, **missing_kwarg_list)

    def test_backend(self):
        for incorrect_backend in EnvironmentTestBase._environment_backends:
            if self.correct_backend == incorrect_backend:
                continue
            os.environ['TRULENS_BACKEND'] = incorrect_backend
            # A None backend is a valid outcome for incorrect backend if imports fail
            if get_backend() is not None:
                self.assertEqual(get_backend().backend, incorrect_backend)
            os.environ['TRULENS_BACKEND'] = self.correct_backend
            self.assertEqual(get_backend().backend, self.correct_backend)
