"""
The TruLens library is designed to support models implemented via a variety of
different popular python neural network frameworks: Keras (with TensorFlow or
Theano backend), TensorFlow, and Pytorch. Models developed with different frameworks
implement things (e.g., gradient computations) a number of different ways. We define
framework specific `ModelWrapper` instances to create a unified model API, providing the same
functionality to models that are implemented in disparate frameworks. In order to compute
attributions for a model, we provide a `trulens.nn.models.get_model_wrapper` function
that will return an appropriate `ModelWrapper` instance.

Some parameters are exclusively utilized for specific frameworks and are outlined
in the parameter descriptions.
"""
import inspect
import os
import traceback

import trulens
from trulens.nn.backend import Backend
from trulens.nn.backend import get_backend
from trulens.utils import tru_logger
from trulens.utils.typing import ModelLike


def discern_backend(model: ModelLike):
    for base_class in inspect.getmro(model.__class__):
        type_str = str(base_class).lower()
        if 'torch' in type_str:
            return Backend.PYTORCH

        else:
            try:
                import tensorflow as tf

                # Graph objects are currently limited to TF1 and Keras backend
                # implies keras backed with TF1 or Theano. TF2 Keras objects are
                # handled by the TF2 backend.
                if 'graph' in type_str:
                    return Backend.TENSORFLOW

                if tf.__version__.startswith('2') and ('tensorflow' in type_str
                                                       or 'keras' in type_str):
                    return Backend.TENSORFLOW

                # Note that in these cases, the TensorFlow version is 1.x.
                elif 'tensorflow' in type_str and 'keras' in type_str and (
                        type_str.index('tensorflow') < type_str.index('keras')):
                    return Backend.TF_KERAS

                elif 'keras' in type_str:
                    return Backend.KERAS

            except (ModuleNotFoundError, ImportError):
                # If tensorflow is not installed, the passed model should not be able to be tensorflow.
                # Note: we can still use Keras without TensorFlow, if the
                #   backend is Theano.
                if 'keras' in type_str:
                    return Backend.KERAS

    return Backend.UNKNOWN


def get_model_wrapper(
    model: ModelLike,
    *,
    logit_layer=None,
    replace_softmax: bool = False,
    softmax_layer=-1,
    custom_objects=None,
    device: str = None,
    input_tensors=None,
    output_tensors=None,
    internal_tensor_dict=None,
    default_feed_dict=None,
    session=None,
    backend=None,
    force_eval=True,
    **kwargs
):
    """
    Returns a ModelWrapper implementation that exposes the components needed for computing attributions.

    Parameters:
        model:
            The model to wrap. If using the TensorFlow 1 backend, this is
            expected to be a graph object.

        logit_layer:
            _Supported for Keras and Pytorch models._
            Specifies the name or index of the layer that produces the
            logit predictions.

        replace_softmax:
            _Supported for Keras models only._ If true, the activation
            function in the softmax layer (specified by `softmax_layer`)
            will be changed to a `'linear'` activation.

        softmax_layer:
            _Supported for Keras models only._ Specifies the layer that
            performs the softmax. This layer should have an `activation`
            attribute. Only used when `replace_softmax` is true.

        custom_objects:
            _Optional, for use with Keras models only._ A dictionary of
            custom objects used by the Keras model.

        device:
            _Optional, for use with Pytorch models only._ A string
            specifying the device to run the model on.

        input_tensors:
            _Required for use with TensorFlow 1 graph models only._ A list
            of tensors representing the input to the model graph.

        output_tensors:
            _Required for use with TensorFlow 1 graph models only._ A list
            of tensors representing the output to the model graph.

        internal_tensor_dict:
            _Optional, for use with TensorFlow 1 graph models only._ A
            dictionary mapping user-selected layer names to the internal
            tensors in the model graph that the user would like to expose.
            This is provided to give more human-readable names to the layers
            if desired. Internal tensors can also be accessed via the name
            given to them by tensorflow.

        default_feed_dict:
            _Optional, for use with TensorFlow 1 graph models only._ A
            dictionary of default values to give to tensors in the model
            graph.

        session:
            _Optional, for use with TensorFlow 1 graph models only._ A
            `tf.Session` object to run the model graph in. If `None`, a new
            temporary session will be generated every time the model is run.

        backend:
            _Optional, for forcing a specific backend._ String values recognized
            are pytorch, tensorflow, keras, or tf.keras.

        force_eval:
            _Optional, True will force a model.eval() call for PyTorch models. False
            will retain current model state

    Returns: ModelWrapper
    """

    if 'input_shape' in kwargs:
        tru_logger.deprecate(
            f'get_model_wrapper: input_shape parameter is no longer used and will be removed in the future'
        )
        del kwargs['input_shape']
    if 'input_dtype' in kwargs:
        tru_logger.deprecate(
            f'get_model_wrapper: input_dtype parameter is no longer used and will be removed in the future'
        )
        del kwargs['input_dtype']

    # get existing backend
    B = get_backend(suppress_warnings=True)

    if backend is None:
        backend = discern_backend(model)
        tru_logger.info(
            'Detected {} backend for {}.'.format(
                backend.name.lower(), type(model)
            )
        )
    else:
        backend = Backend.from_name(backend)
    if B is None or (backend is not Backend.UNKNOWN and B.backend != backend):
        tru_logger.info(
            'Changing backend from {} to {}.'.format(
                None if B is None else B.backend, backend
            )
        )
        os.environ['TRULENS_BACKEND'] = backend.name.lower()
        B = get_backend()
    else:
        tru_logger.info('Using backend {}.'.format(B.backend))
    tru_logger.info(
        'If this seems incorrect, you can force the correct backend by passing the `backend` parameter directly into your get_model_wrapper call.'
    )
    if B.backend.is_keras_derivative():
        from trulens.nn.models.keras import KerasModelWrapper
        return KerasModelWrapper(
            model,
            logit_layer=logit_layer,
            replace_softmax=replace_softmax,
            softmax_layer=softmax_layer,
            custom_objects=custom_objects
        )

    elif B.backend == Backend.PYTORCH:
        from trulens.nn.models.pytorch import PytorchModelWrapper
        return PytorchModelWrapper(
            model,
            logit_layer=logit_layer,
            device=device,
            force_eval=force_eval
        )
    elif B.backend == Backend.TENSORFLOW:
        import tensorflow as tf
        if tf.__version__.startswith('2'):
            from trulens.nn.models.tensorflow_v2 import Tensorflow2ModelWrapper
            return Tensorflow2ModelWrapper(
                model,
                logit_layer=logit_layer,
                replace_softmax=replace_softmax,
                softmax_layer=softmax_layer,
                custom_objects=custom_objects
            )
        else:
            from trulens.nn.models.tensorflow_v1 import TensorflowModelWrapper
            if input_tensors is None:
                tru_logger.error(
                    'tensorflow1 model must pass parameter: input_tensors'
                )
            if output_tensors is None:
                tru_logger.error(
                    'tensorflow1 model must pass parameter: output_tensors'
                )
            return TensorflowModelWrapper(
                model,
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                internal_tensor_dict=internal_tensor_dict,
                session=session
            )
