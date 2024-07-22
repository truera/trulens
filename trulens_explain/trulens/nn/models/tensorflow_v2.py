from typing import Tuple

import tensorflow as tf
from trulens.nn.backend import get_backend
from trulens.nn.models.keras import \
    KerasModelWrapper  # dangerous to have this here if tf-less keras gets imported
from trulens.nn.quantities import QoI
from trulens.nn.slices import Cut
from trulens.nn.slices import InputCut
from trulens.nn.slices import LogitCut
from trulens.nn.slices import OutputCut
from trulens.utils import tru_logger
from trulens.utils.typing import DATA_CONTAINER_TYPE
from trulens.utils.typing import Inputs
from trulens.utils.typing import many_of_om
from trulens.utils.typing import ModelInputs
from trulens.utils.typing import nested_cast
from trulens.utils.typing import nested_map
from trulens.utils.typing import om_of_many
from trulens.utils.typing import Outputs
from trulens.utils.typing import TensorArgs
from trulens.utils.typing import TensorLike

if tf.executing_eagerly():
    tf.config.run_functions_eagerly(True)


class Tensorflow2ModelWrapper(KerasModelWrapper
                             ):  # dangerous to extend keras model wrapper
    """
    Model wrapper that exposes internal layers of tf2 Keras models.
    """

    def __init__(
        self,
        model,
        *,
        logit_layer=None,
        replace_softmax=False,
        softmax_layer=-1,
        custom_objects=None,
        **kwargs
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
            custom_objects=custom_objects,
            **kwargs
        )

        self._eager = tf.executing_eagerly()

        # In eager mode, we have to use hook functions to get intermediate
        # outputs and internal gradients.
        # See: https://github.com/tensorflow/tensorflow/issues/33478
        if self._eager:
            self._warn_keras_layers(self._layers.values())

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

            for layer in self._layers.values():
                layer.call = get_call_fn(layer)

            self._cached_input = []

        self._B = get_backend()

    @property
    def B(self):
        return self._B

    def _warn_keras_layers(self, layers):
        keras_layers = [l for l in layers if 'KerasLayer' in str(type(l))]
        if (keras_layers):
            tru_logger.warning('Detected a KerasLayer in the model. This can sometimes create issues during attribution runs or subsequent model calls. \
                If failures occur: try saving the model, deactivating eager mode with tf.compat.v1.disable_eager_execution(), \
                setting tf.config.run_functions_eagerly(False), and reloading the model.'\
                    'Detected KerasLayers from model.layers: %s' % str(keras_layers))

    def _clear_hooks(self):
        for layer in self._layers.values():
            layer.input_intervention = None
            layer.output_intervention = None
            layer.retrieve_functions = []

    def _get_output_layer(self):
        output_layers = []
        if self._model.outputs is None:
            raise Exception(
                'Unable to determine output layers. Please set the outputs using set_output_layers.'
            )
        for output in self._model.outputs:
            for layer in self._layers.values():
                try:
                    if layer is output or self._get_layer_output(layer
                                                                ) is output:
                        output_layers.append(layer)
                except:
                    # layer output may not be instantiated when using model subclassing,
                    # but it is not a problem because self._model.outputs is only autoselected as
                    # the output_layer output when not subclassing.
                    continue

        return output_layers

    def _is_input_layer(self, layer):
        if (self._model.inputs is not None):
            return any(
                [
                    inpt is self._get_layer_output(layer)
                    for inpt in self._model.inputs
                ]
            )
        else:
            return False

    def _input_layer_index(self, layer):
        for i, inpt in enumerate(self._model.inputs):
            if inpt is self._get_layer_output(layer):
                return i

        return None

    def set_output_layers(self, output_layers: list):
        if not isinstance(output_layers, list):
            raise Exception('Output Layers must be a list of layers')
        self._model.outputs = output_layers

    def _fprop(
        self, *, model_inputs: ModelInputs, doi_cut: Cut, to_cut: Cut,
        attribution_cut: Cut, intervention: TensorArgs
    ) -> Tuple[Outputs[TensorLike], Outputs[TensorLike]]:
        """
        See ModelWrapper.fprop .
        """

        B = get_backend()

        if not self._eager:
            return super()._fprop(
                model_inputs=model_inputs,
                doi_cut=doi_cut,
                to_cut=to_cut,
                attribution_cut=attribution_cut,
                intervention=intervention
            )

        attribution_results = None

        try:
            if intervention:
                if not isinstance(doi_cut, InputCut):
                    from_layers = (
                        self._get_logit_layer()
                        if isinstance(doi_cut,
                                      LogitCut) else self._get_output_layer()
                        if isinstance(doi_cut, OutputCut) else
                        self._get_layers_by_name(doi_cut.name)
                    )

                    for layer, x_i in zip(from_layers, intervention.args):

                        def intervention_fn(x):
                            nonlocal x_i
                            x, x_i = many_of_om(x), many_of_om(x_i)
                            for i, (_x, _x_i) in enumerate(zip(x, x_i)):
                                if _x.dtype != _x_i.dtype:
                                    x_i[i] = tf.cast(_x_i, _x.dtype)
                            return om_of_many(x_i)

                        if doi_cut.anchor == 'in':
                            layer.input_intervention = intervention_fn
                        else:
                            if doi_cut.anchor is not None and doi_cut.anchor != 'out':
                                tru_logger.warning(
                                    f'Unrecognized doi_cut.anchor {doi_cut.anchor}. Defaulting to `out` anchor.'
                                )
                            layer.output_intervention = intervention_fn
                else:
                    model_inputs = intervention

            # Get the output from the "to layers" and possibly the latent
            # layers.
            def retrieve_index(i, results, anchor):

                def retrieve(inputs, output):
                    if anchor == 'in':
                        # Why does this happen:?
                        if isinstance(inputs,
                                      DATA_CONTAINER_TYPE) and len(inputs) == 1:
                            results[i] = inputs[0]
                        else:
                            results[i] = inputs
                    else:
                        results[i] = output

                return retrieve

            results: Outputs[TensorLike]

            # TODO: Clean this all up somehow: trulens for TF2 allows for cuts
            # with anchors that can refer to a layers's inputs or outputs.
            # Layers can have more than 1 input. In those cases, the size of
            # attribution_layers is not indicative of the how many tensors there
            # will be in the results of this call.

            if isinstance(to_cut, InputCut):
                results: Outputs[TensorLike] = model_inputs.args

            else:
                if isinstance(to_cut, LogitCut):
                    to_layers = self._get_logit_layer()
                elif isinstance(to_cut, OutputCut):
                    to_layers = self._get_output_layer()
                else:
                    to_layers = self._get_layers_by_name(to_cut.name)

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
                    if isinstance(attribution_cut, LogitCut):
                        attribution_layers = self._get_logit_layer()
                    elif isinstance(attribution_cut, OutputCut):
                        attribution_layers = self._get_output_layer()
                    else:
                        attribution_layers = self._get_layers_by_name(
                            attribution_cut.name
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
            # keras.Layer have similar argument handling to our public interfaces
            self._model(om_of_many(model_inputs.args))

        finally:
            # Clear the hooks after running the model so that `fprop` doesn't
            # leave the model in an altered state.
            self._clear_hooks()

        results = self._flatten(results)

        if attribution_results is not None:
            attribution_results = self._flatten(attribution_results)

        return (results, attribution_results)

    def _qoi_bprop(
        self, *, qoi: QoI, model_inputs: ModelInputs, doi_cut: Cut, to_cut: Cut,
        attribution_cut: Cut, intervention: TensorArgs
    ) -> Outputs[Inputs[TensorLike]]:
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
            intervention = intervention.map(tf.constant)

            intervention.foreach(tape.watch)

            outputs, attribution_features = self._fprop(
                model_inputs=model_inputs,
                doi_cut=doi_cut,
                to_cut=to_cut,
                attribution_cut=attribution_cut,
                intervention=intervention
            )
            outputs: Outputs[TensorLike]
            attribution_features: Outputs[TensorLike]

            Q = qoi._wrap_public_call(outputs)

        grads: Outputs[Inputs[TensorLike]] = []

        for z in attribution_features:
            zq: Inputs[TensorLike] = []
            for q in Q:
                grad_zq = tape.gradient(q, z)
                zq.append(grad_zq)

            grads.append(zq)

        grads = nested_map(grads, attribution_cut.access_layer)

        del tape

        grads = list(zip(*grads))  # transpose

        return grads
