import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from netlens import backend as B
from netlens.models.keras import KerasModelWrapper
from netlens.models._model_base import ModelWrapper
from netlens.slices import InputCut, OutputCut, LogitCut


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
            custom_objects=None):
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
            custom_objects=custom_objects)

        self._eager = tf.executing_eagerly()

        # In eager mode, we have to use hook functions to get intermediate
        # outputs and internal gradients.
        # See: https://github.com/tensorflow/tensorflow/issues/33478
        if self._eager:

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

        self._B = B

    @property
    def B(self):
        return self._B

    def _clear_hooks(self):
        for layer in self._layers:
            layer.input_intervention = None
            layer.output_intervention = None
            layer.retrieve_functions = []

    def _get_output_layer(self):
        output_layers = []
        for output in self._model.outputs:
            for layer in self._layers:
                if layer.output is output:
                    output_layers.append(layer)

        return output_layers

    def _is_input_layer(self, layer):
        return any([inpt is layer.output for inpt in self._model.inputs])

    def _input_layer_index(self, layer):
        for i, inpt in enumerate(self._model.inputs):
            if inpt is layer.output:
                return i

        return None

    def fprop(self, x, from_cut=None, to_cut=None, latent_cut=None):
        """
        fprop Forward propagate the model

        Parameters
        ----------
        x : backend.Tensor or np.array
            Input tensor to propagate through the model. If an np.array, 
            will be converted to a tensor on the same device as the model.
        from_cut: Cut, optional
            The Cut from which to begin propagation. The shape of `x` must
            match the input shape of this layer. By default 0.
        to_cut : Cut, optional
            The Cut to return output activation tensors for. If None,
            assumed to be just the final layer. By default None
        latent_cut : Cut, optional
            An additional Cut to return activation tensors for. This is
            usually used to apply distributions of interest (DoI)

        Returns
        -------
        (list of backend.Tensor)
            A list of output activations are returned.
        if `latent_cut` is supplied, also return the latent cut activations
        """
        if not self._eager:
            return super().fprop(x, from_cut, to_cut)

        if from_cut is None:
            from_cut = InputCut()
        if to_cut is None:
            to_cut = OutputCut()

        if not isinstance(x, list):
            x = [x]

        # We return a numpy array if we were given a numpy array; otherwise we
        # will let the returned values remain data tensors.
        return_numpy = isinstance(x[0], np.ndarray)

        # Convert `x` to a data tensor if it isn't already.
        if return_numpy:
            x = ModelWrapper._nested_apply(x, tf.constant)

        try:
            if not isinstance(from_cut, InputCut):
                # Get Inputs and batch then the same as DoI resolution
                if (self._cached_input):
                    model_input = self._cached_input

                    doi_repeated_batch_size = x[0].shape[0]
                    batched_model_input = []
                    for val in model_input:
                        if isinstance(val, np.ndarray):
                            doi_resolution = int(
                                doi_repeated_batch_size / val.shape[0])
                            tile_shape = [1] * len(val.shape)
                            tile_shape[0] = doi_resolution
                            val = np.tile(val, tuple(tile_shape))
                        elif tf.is_tensor(val):
                            doi_resolution = int(
                                doi_repeated_batch_size / val.shape[0])
                            val = tf.repeat(val, doi_resolution, axis=0)
                        batched_model_input.append(val)
                    model_input = batched_model_input
                else:
                    model_input = x

                from_layers = (
                    self._get_logit_layer() if (isinstance(
                        from_cut, LogitCut)) else self._get_output_layer() if
                    (isinstance(from_cut, OutputCut)) else
                    self._get_layers_by_name(from_cut.name))

                for layer, x_i in zip(from_layers, x):
                    if from_cut.anchor == 'in':
                        layer.input_intervention = lambda _: x_i
                    else:
                        layer.output_intervention = lambda _: x_i
            else:
                model_input = x
                self._cached_input = model_input

            # Get the output from the "to layers," and possibly the latent
            # layers.
            def retrieve_index(i, results, anchor):

                def retrieve(inputs, output):
                    if anchor == 'in':
                        results[i] = (
                            inputs[0] if (
                                (
                                    isinstance(inputs, list) or
                                    isinstance(inputs, tuple)) and
                                len(inputs) == 1) else inputs)
                    else:
                        results[i] = (
                            output[0] if
                            (isinstance(output, list) and
                             len(output) == 1) else output)

                return retrieve

            if isinstance(to_cut, InputCut):
                results = x

            else:
                to_layers = (
                    self._get_logit_layer() if (isinstance(
                        to_cut, LogitCut)) else self._get_output_layer() if
                    (isinstance(to_cut, OutputCut)) else
                    self._get_layers_by_name(to_cut.name))

                results = [None for _ in to_layers]

                for i, layer in enumerate(to_layers):
                    layer.retrieve_functions.append(
                        retrieve_index(i, results, to_cut.anchor))

            if latent_cut:
                if isinstance(latent_cut, InputCut):
                    latent_results = x

                else:
                    latent_layers = (
                        self._get_logit_layer() if
                        (isinstance(latent_cut,
                                    LogitCut)) else self._get_output_layer() if
                        (isinstance(latent_cut, OutputCut)) else
                        self._get_layers_by_name(latent_cut.name))

                    latent_results = [None for _ in latent_layers]

                    for i, layer in enumerate(latent_layers):
                        if self._is_input_layer(layer):
                            # Input layers don't end up calling the hook, so we
                            # have to get their output manually.
                            latent_results[i] = x[self._input_layer_index(
                                layer)]

                        else:
                            layer.retrieve_functions.append(
                                retrieve_index(
                                    i, latent_results, latent_cut.anchor))

            # Run a point.
            self._model(model_input)

        finally:
            # Clear the hooks after running the model so that `fprop` doesn't
            # leave the model in an altered state.
            self._clear_hooks()

        if return_numpy:
            results = ModelWrapper._nested_apply(results, lambda t: t.numpy())

        return (results, latent_results) if latent_cut else results

    def qoi_bprop(self, x, qoi, from_cut=None, to_cut=None, doi_cut=None):
        """
        qoi_bprop Run the model from the from_layer to the qoi layer
            and give the gradients w.r.t x

        Parameters
        ----------
        x : backend.Tensor or np.array
            Input tensor to propagate through the model. If an np.array,
            will be converted to a tensor on the same device as the model.
        qoi: a Quantity of Interest
            This method will accumulate all gradients of the qoi w.r.t x
        from_cut: Cut, optional
            if `from_cut` is None, this refers to the InputCut.
            The Cut in which attribution will be calculated. This is generally
            taken from the attribution slyce's from_cut.
        to_cut: Cut, optional
            if `to_cut` is None, this refers to the OutputCut.
            The Cut in which qoi will be calculated. This is generally
            taken from the attribution slyce's to_cut.
        doi_cut: Cut, 
            if `doi_cut` is None, this refers to the InputCut.
            Cut from which to begin propagation. The shape of `x` must
            match the output shape of this layer.

        Returns
        -------
        np.array of the gradients of `qoi` w.r.t. `from_cut`
        """
        x = x if isinstance(x, list) else [x]
        if not self._eager:
            return super().qoi_bprop(x, qoi, from_cut, to_cut, doi_cut)

        if from_cut is None:
            from_cut = InputCut()
        if to_cut is None:
            to_cut = OutputCut()

        # We return a numpy array if we were given a numpy array; otherwise we
        # will let the returned values remain data tensors.
        return_numpy = isinstance(x, np.ndarray) or isinstance(x[0], np.ndarray)

        # Convert `x` to a data tensor if it isn't already.
        if return_numpy:
            x = [ModelWrapper._nested_apply(x_i, tf.constant) for x_i in x]

        with tf.GradientTape(persistent=True) as tape:
            for x_i in x:
                ModelWrapper._nested_apply(x_i, tape.watch)

            outputs, latent_features = self.fprop(
                x,
                doi_cut if doi_cut else InputCut(),
                to_cut,
                latent_cut=from_cut)
            Q = qoi(outputs[0]) if len(outputs) == 1 else qoi(outputs)
            if isinstance(Q, list) and len(Q) == 1:
                Q = B.sum(Q)

        grads = [tape.gradient(q, latent_features) for q in Q] if isinstance(
            Q, list) else tape.gradient(Q, latent_features)
        grads = (
            grads[0] if isinstance(grads, list) and len(grads) == 1 else grads)
        grads = [from_cut.access_layer(g) for g in grads] if isinstance(
            grads, list) else from_cut.access_layer(grads)

        del tape

        if return_numpy:
            grads = [ModelWrapper._nested_apply(g, B.as_array) for g in grads] \
                if isinstance(grads, list) else \
            ModelWrapper._nested_apply(grads, B.as_array)

        return (
            grads[0] if isinstance(grads, list) and len(grads) == 1 else grads)
