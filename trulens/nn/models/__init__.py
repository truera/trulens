import os

import trulens
from trulens.utils import tru_logger
from trulens.nn.backend import get_backend

def discern_backend(model):
    type_str = str(type(model)).lower()
    
    if 'torch' in type_str:
        return 'pytorch'
    else:
        import tensorflow as tf
        # graph objects are currently limited to TF1 and Keras backend implies keras backed with TF1 or Theano.
        # TF2 Keras objects are handled by the TF2 backend
        if 'graph' in type_str or tf.__version__.startswith('2'):
            return 'tensorflow'
        elif 'tf.keras' in type_str:
            return 'tf.keras'
        else:
            return 'keras'


def get_model_wrapper(
            model,
            logit_layer=None,
            replace_softmax=False,
            softmax_layer=-1,
            custom_objects=None,
            input_shape=None,
            device=None,
            input_tensors=None,
            output_tensors=None,
            internal_tensor_dict=None,
            default_feed_dict=None,
            session=None,
            backend=None):
    """
    Returns a ModelWrapper implementation that exposes the components needed for computing attributions.

    Parameters:
        model:
            The model to wrap. For the TensorFlow 1 backend, this is 
            expected to be a graph object.

        logit_layer:
            Specifies the name or index of the layer that produces the
            logit predictions. Supported for Keras and Pytorch models.

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

        input_shape:
            _Required for use with Pytorch models only._ Tuple specifying
            the input shape (excluding the batch dimension) expected by the
            model.

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
            are pytorch, tensorflow/tf, keras, or tf.keras.
    """
    if backend is None:
        backend = discern_backend(model)
        tru_logger.info("Detected {} backend for {}".format(backend, type(model)))
    B = get_backend()
    if get_backend().backend != backend:
        tru_logger.info("Changing backend from {} to {}".format(get_backend().backend, backend))
        os.environ['TRULENS_BACKEND'] = backend
        B = get_backend()
    
    if get_backend().backend == 'keras' or get_backend().backend == 'tf.keras':
        from trulens.nn.models.keras import KerasModelWrapper
        return KerasModelWrapper(model,
            logit_layer=logit_layer,
            replace_softmax=replace_softmax,
            softmax_layer=softmax_layer,
            custom_objects=custom_objects)

    elif get_backend().backend == 'pytorch':
        from trulens.nn.models.pytorch import PytorchModelWrapper
        if input_shape is None:
            tru_logger.error('pytorch model must pass parameter: input_shape')
        return PytorchModelWrapper(model,
            input_shape,
            input_dtype=input_dtype,
            logit_layer=logit_layer,
            device=device)

    elif get_backend().backend == 'tensorflow':
        import tensorflow as tf

        if tf.__version__.startswith('2'):
            from trulens.nn.models.tensorflow_v2 import Tensorflow2ModelWrapper
            return Tensorflow2ModelWrapper(model,
                logit_layer=logit_layer,
                replace_softmax=replace_softmax,
                softmax_layer=softmax_layer,
                custom_objects=custom_objects)
        else:
            from trulens.nn.models.tensorflow_v1 import TensorflowModelWrapper
            if input_tensors is None:
                tru_logger.error('tensorflow1 model must pass parameter: input_tensors')
            if output_tensors is None:
                tru_logger.error('tensorflow1 model must pass parameter: output_tensors') 
            return TensorflowModelWrapper(model,
                input_tensors,
                output_tensors,
                internal_tensor_dict=internal_tensor_dict,
                session=session)

    