import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from trulens.nn.backend import get_backend
from trulens.nn.models._model_base import ModelWrapper
from trulens.nn.models.keras import KerasModelWrapper
from trulens.nn.quantities import QoI
from trulens.nn.slices import Cut, InputCut
from trulens.nn.slices import LogitCut
from trulens.nn.slices import OutputCut
from trulens.utils import tru_logger
from trulens.utils.typing import DATA_CONTAINER_TYPE, ModelInputs

if tf.executing_eagerly():
    tf.config.run_functions_eagerly(True)


class Tensorflow2ModelWrapper(KerasModelWrapper):
    """
    Model wrapper that exposes internal layers of tf2 Keras models.
    """

    def __init__(
        self,
        model,
        logit_layer=None,
        replace_softmax=False,
        softmax_layer=-1,
        custom_objects=None
    ):
        """
        __init__ Constructor

        Parameters
        ----------
        model : tf.keras.Model
            tf.keras.Model or a subclass
        eager: bool, optional:
            whether or not model is in eager mode.
        """
        super().__init__(
            model,
            logit_layer=logit_layer,
            replace_softmax=replace_softmax,
            softmax_layer=softmax_layer,
            custom_objects=custom_objects
        )

        self._eager = tf.executing_eagerly()

        # In eager mode, we have to use hook functions to get intermediate
        # outputs and internal gradients.
        # See: https://github.com/tensorflow/tensorflow/issues/33478
        if self._eager:
            self._warn_keras_layers(self._layers)

            def get_call_fn(layer):
                old_call_fn = layer.call

                def call(*inputs, **kwargs):
                    if layer.input_intervention:
                        inputs = [layer.input_intervention(inputs)]

                    output = old_call_fn(*inputs, **kwargs)

                    if layer.output_intervention:
                        output = layer.output_intervention(output)

                    for retrieve in layer.retrieve_functions:
                        retrieve(inputs, output)

                    return output

                return call

            self._clear_hooks()

            for layer in self._layers:
                layer.call = get_call_fn(layer)

            self._cached_input = []

        self._B = get_backend()

    @property
    def B(self):
        return self._B

    def _warn_keras_layers(self, layers):
        keras_layers = [l for l in layers if 'KerasLayer' in str(type(l))]
        if (keras_layers):
            tru_logger.warn('Detected a KerasLayer in the model. This can sometimes create issues during attribution runs or subsequent model calls. \
                If failures occur: try saving the model, deactivating eager mode with tf.compat.v1.disable_eager_execution(), \
                setting tf.config.run_functions_eagerly(False), and reloading the model.'\
                    'Detected KerasLayers from model.layers: %s' % str(keras_layers))

    def _clear_hooks(self):
        for layer in self._layers:
            layer.input_intervention = None
            layer.output_intervention = None
            layer.retrieve_functions = []

    def _get_output_layer(self):
        output_layers = []
        if self._model.outputs is None:
            raise Exception(
                "Unable to determine output layers. Please set the outputs using set_output_layers."
            )
        for output in self._model.outputs:
            for layer in self._layers:
                try:
                    if layer is output or layer.output is output:
                        output_layers.append(layer)
                except:
                    # layer.output may not be instantiated when using model subclassing,
                    # but it is not a problem because self._model.outputs is only autoselected as output_layer.output
                    # when not subclassing.
                    continue

        return output_layers

    def _is_input_layer(self, layer):
        if (self._model.inputs is not None):
            return any([inpt is layer.output for inpt in self._model.inputs])
        else:
            return False

    def _input_layer_index(self, layer):
        for i, inpt in enumerate(self._model.inputs):
            if inpt is layer.output:
                return i

        return None

    def set_output_layers(self, output_layers: list):
        if not isinstance(output_layers, list):
            raise Exception("Output Layers must be a list of layers")
        self._model.outputs = output_layers

    def _fprop(
        self,
        *,
        model_inputs: ModelInputs,
        doi_cut: Cut,
        to_cut: Cut,
        attribution_cut: Cut,
        intervention: ModelInputs
    ):
        """
        See ModelWrapper.fprop .
        """

        if not self._eager:
            return super()._fprop(
                model_inputs=model_inputs,
                doi_cut=doi_cut,
                to_cut=to_cut,
                attribution_cut=attribution_cut,
                intervention=intervention
            )

        # model_args = model_inputs.args
        # model_kwargs = model_inputs.kwargs
        # intervention = intervention.args

        # if isinstance(doi_cut, InputCut):  # This logic happens later in this backend.
        #     model_args = intervention

        return_numpy = True

        if intervention is not None:
            # if not isinstance(intervention, DATA_CONTAINER_TYPE):
            #     intervention = [intervention]

            # We return a numpy array if we were given a numpy array; otherwise
            # we will let the returned values remain data tensors.
            return_numpy = isinstance(intervention.first(), np.ndarray)

            assert not return_numpy

            # Convert `x` to a data tensor if it isn't already.
            #if return_numpy:
            #    intervention = intervention.map(lambda t: ModelWrapper._nested_apply(t, tf.constant))
            

        try:
            if intervention:
                # Get Inputs and batch then the same as DoI resolution
                doi_repeated_batch_size = intervention.first().shape[0]

                batched_model_args = []

                for val in model_inputs.args:
                    if isinstance(val, np.ndarray):
                        doi_resolution = int(
                            doi_repeated_batch_size / val.shape[0]
                        )
                        tile_shape = [1] * len(val.shape)
                        tile_shape[0] = doi_resolution
                        val = B.as_tensor(np.tile(val, tuple(tile_shape)))

                    elif tf.is_tensor(val):
                        doi_resolution = int(
                            doi_repeated_batch_size / val.shape[0]
                        )
                        val = tf.repeat(val, doi_resolution, axis=0)

                    batched_model_args.append(val)

                model_inputs = ModelInputs(batched_model_args, model_inputs.kwargs)

                if not isinstance(doi_cut, InputCut):
                    from_layers = (
                        self._get_logit_layer()
                        if isinstance(doi_cut,
                                      LogitCut) else self._get_output_layer()
                        if isinstance(doi_cut, OutputCut) else
                        self._get_layers_by_name(doi_cut.name)
                    )

                    for layer, x_i in zip(from_layers, intervention.args):
                        if doi_cut.anchor == 'in':
                            layer.input_intervention = lambda _: x_i
                        else:
                            layer.output_intervention = lambda _: x_i
                else:
                    #arg_wrapped_list = False
                    # Take care of the Keras Module case where args is a tuple
                    # of list of inputs corresponding to `model._inputs`. This
                    # would have gotten unwrapped as the logic operates on the
                    # list of inputs. so needs to be re-wrapped in tuple for the
                    # model arg execution.
                    #if (isinstance(model_inputs.args, DATA_CONTAINER_TYPE) and
                    #        isinstance(model_inputs.args[0], DATA_CONTAINER_TYPE)):

                    #    arg_wrapped_list = True

                    model_inputs = intervention

                    #if arg_wrapped_list:
                    #    model_args = (model_args,)

            # Get the output from the "to layers," and possibly the latent
            # layers.
            def retrieve_index(i, results, anchor):

                def retrieve(inputs, output):
                    if anchor == 'in':
                        results[i] = (
                            inputs[0] if (
                                isinstance(inputs, DATA_CONTAINER_TYPE) and
                                len(inputs) == 1
                            ) else inputs
                        )
                    else:
                        results[i] = (
                            output[0] if (
                                isinstance(output, DATA_CONTAINER_TYPE) and
                                len(output) == 1
                            ) else output
                        )

                return retrieve

            if isinstance(to_cut, InputCut):
                results = model_inputs.args

            else:
                to_layers = (
                    self._get_logit_layer() if (isinstance(to_cut, LogitCut))
                    else self._get_output_layer() if
                    (isinstance(to_cut, OutputCut)) else
                    self._get_layers_by_name(to_cut.name)
                )

                results = [None for _ in to_layers]

                for i, layer in enumerate(to_layers):
                    layer.retrieve_functions.append(
                        retrieve_index(i, results, to_cut.anchor)
                    )

            if attribution_cut:
                if isinstance(attribution_cut, InputCut):
                    # The attribution must be the watched tensor given from
                    # `qoi_bprop`.
                    attribution_results = intervention.args

                else:
                    attribution_layers = (
                        self._get_logit_layer() if
                        (isinstance(attribution_cut,
                                    LogitCut)) else self._get_output_layer() if
                        (isinstance(attribution_cut, OutputCut)) else
                        self._get_layers_by_name(attribution_cut.name)
                    )

                    attribution_results = [None for _ in attribution_layers]

                    for i, layer in enumerate(attribution_layers):
                        if self._is_input_layer(layer):
                            # Input layers don't end up calling the hook, so we
                            # have to get their output manually.
                            attribution_results[i] = intervention.args[
                                self._input_layer_index(layer)]

                        else:
                            layer.retrieve_functions.append(
                                retrieve_index(
                                    i, attribution_results,
                                    attribution_cut.anchor
                                )
                            )

            # Run a point.
            # Some Models require inputs as single tensors while others require a list.
            if len(model_inputs.args) == 1:
                model_args = model_inputs.args[0]
            else:
                model_args = model_inputs.args

            self._model(model_args)

        finally:
            # Clear the hooks after running the model so that `fprop` doesn't
            # leave the model in an altered state.
            self._clear_hooks()

        #if return_numpy:
        #    results = ModelWrapper._nested_apply(
        #        results, lambda t: t.numpy()
        #        if not isinstance(t, np.ndarray) else t
        #    )

        print("results=", results)

        return (results, attribution_results) if attribution_cut else results

    def _qoi_bprop(
        self,
        *,
        qoi: QoI,
        model_inputs: ModelInputs,
        doi_cut: Cut,
        to_cut: Cut,
        attribution_cut: Cut,
        intervention: ModelInputs
    ):
        """
        See ModelWrapper.qoi_bprop .
        """

        if not self._eager:
            return super()._qoi_bprop(
                qoi=qoi,
                model_inputs=model_inputs,
                doi_cut=doi_cut,
                to_cut=to_cut,
                attribution_cut=attribution_cut,
                intervention=intervention
            )

        with tf.GradientTape(persistent=True) as tape:
            # We return a numpy array if we were given a numpy array; otherwise
            # we will let the returned values remain data tensors.

            return_numpy = isinstance(intervention.first(), np.ndarray)

            # Convert `intervention` to a data tensor if it isn't already.

            if return_numpy:
                intervention = intervention.map(lambda t: ModelWrapper._nested_apply(t, tf.constant))
            
            intervention.foreach(lambda t: ModelWrapper._nested_apply(t, tape.watch))

            outputs, attribution_features = self._fprop(
                model_inputs=model_inputs,
                doi_cut=doi_cut,
                to_cut=to_cut,
                attribution_cut=attribution_cut,
                intervention=intervention
            )
            if isinstance(outputs, DATA_CONTAINER_TYPE) and isinstance(
                    outputs[0], DATA_CONTAINER_TYPE):
                outputs = outputs[0]

            Q = qoi(outputs[0]) if len(outputs) == 1 else qoi(outputs)
            if isinstance(Q, DATA_CONTAINER_TYPE) and len(Q) == 1:
                Q = get_backend().sum(Q)

        grads = [tape.gradient(q, attribution_features) for q in Q
                ] if isinstance(Q, DATA_CONTAINER_TYPE
                               ) else tape.gradient(Q, attribution_features)

        grads = grads[0] if isinstance(grads, DATA_CONTAINER_TYPE
                                      ) and len(grads) == 1 else grads

        grads = [attribution_cut.access_layer(g) for g in grads] if isinstance(
            grads, DATA_CONTAINER_TYPE
        ) else attribution_cut.access_layer(grads)

        del tape

        #if return_numpy:
        #    grads = [
        #        ModelWrapper._nested_apply(g,
        #                                   get_backend().as_array)
        #        for g in grads
        #    ] if isinstance(
        #        grads, DATA_CONTAINER_TYPE
        #    ) else ModelWrapper._nested_apply(grads,
        #                                      get_backend().as_array)

        return grads

        # return grads[0] if isinstance(grads, DATA_CONTAINER_TYPE
        #                             ) and len(grads) == 1 else grads
