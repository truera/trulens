import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from trulens.nn import backend as B
from trulens.nn.models.keras import KerasModelWrapper
from trulens.nn.models._model_base import ModelWrapper, DATA_CONTAINER_TYPE
from trulens.nn.slices import InputCut, OutputCut, LogitCut


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
        if (self._model.inputs is not None):
            return any([inpt is layer.output for inpt in self._model.inputs])
        else:
            return False

    def _input_layer_index(self, layer):
        for i, inpt in enumerate(self._model.inputs):
            if inpt is layer.output:
                return i

        return None

    def fprop(
            self,
            model_args,
            model_kwargs={},
            doi_cut=None,
            to_cut=None,
            attribution_cut=None,
            intervention=None):
        """
        fprop Forward propagate the model

        Parameters
        ----------
        model_args, model_kwargs: 
            The args and kwargs given to the call method of a model.
            This should represent the instances to obtain attributions for, 
            assumed to be a *batched* input. if `self.model` supports evaluation 
            on *data tensors*, the  appropriate tensor type may be used (e.g., 
            Pytorch models may accept Pytorch tensors in addition to 
            `np.ndarray`s). The shape of the inputs must match the input shape
            of `self.model`.
        doi_cut: Cut, optional
            The Cut from which to begin propagation. The shape of `intervention`
            must match the input shape of this layer. This is usually used to 
            apply distributions of interest (DoI)
        to_cut : Cut, optional
            The Cut to return output activation tensors for. If `None`,
            assumed to be just the final layer. By default None
        attribution_cut : Cut, optional
            An Cut to return activation tensors for. If `None` 
            attributions layer output is not returned.
        intervention : backend.Tensor or np.array
            Input tensor to propagate through the model. If an np.array, 
            will be converted to a tensor on the same device as the model.

        Returns
        -------
        (list of backend.Tensor or np.ndarray)
            A list of output activations are returned, preferring to stay in the
            same format as the input. If `attribution_cut` is supplied, also 
            return the cut activations.
        """
        if not self._eager:
            return super().fprop(
                model_args, 
                model_kwargs, 
                doi_cut, 
                to_cut, 
                attribution_cut,
                intervention)

        if doi_cut is None:
            doi_cut = InputCut()
        if to_cut is None:
            to_cut = OutputCut()

        return_numpy = True

        if intervention is not None:
            if not isinstance(intervention, DATA_CONTAINER_TYPE):
                intervention = [intervention]

            # We return a numpy array if we were given a numpy array; otherwise
            # we will let the returned values remain data tensors.
            return_numpy = isinstance(intervention[0], np.ndarray)

            # Convert `x` to a data tensor if it isn't already.
            if return_numpy:
                intervention = ModelWrapper._nested_apply(
                    intervention, tf.constant)

        try:
            if (intervention):
                # Get Inputs and batch then the same as DoI resolution
                doi_repeated_batch_size = intervention[0].shape[0]
                batched_model_args = []
                for val in model_args:
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
                    batched_model_args.append(val)
                model_args = batched_model_args

                if not isinstance(doi_cut, InputCut):
                    from_layers = (
                        self._get_logit_layer() if isinstance(
                            doi_cut, LogitCut) else 
                        self._get_output_layer() if isinstance(
                            doi_cut, OutputCut) else
                        self._get_layers_by_name(doi_cut.name))

                    for layer, x_i in zip(from_layers, intervention):
                        if doi_cut.anchor == 'in':
                            layer.input_intervention = lambda _: x_i
                        else:
                            layer.output_intervention = lambda _: x_i
                else:
                    arg_wrapped_list = False
                    # Take care of the Keras Module case where args is a tuple 
                    # of list of inputs corresponding to `model._inputs`. This
                    # would have gotten unwrapped as the logic operates on the
                    # list of inputs. so needs to be re-wrapped in tuple for the
                    # model arg execution.
                    if (isinstance(model_args, DATA_CONTAINER_TYPE) and 
                            isinstance(model_args[0], DATA_CONTAINER_TYPE)):

                        arg_wrapped_list = True

                    model_args = intervention

                    if arg_wrapped_list:
                        model_args = (model_args,)

            # Get the output from the "to layers," and possibly the latent
            # layers.
            def retrieve_index(i, results, anchor):

                def retrieve(inputs, output):
                    if anchor == 'in':
                        results[i] = (
                            inputs[0] if (
                                isinstance(inputs, DATA_CONTAINER_TYPE) and
                                len(inputs) == 1) else inputs)
                    else:
                        results[i] = (
                            output[0] if (
                                isinstance(output, DATA_CONTAINER_TYPE) and
                                len(output) == 1) else output)

                return retrieve

            if isinstance(to_cut, InputCut):
                results = model_args

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

            if attribution_cut:
                if isinstance(attribution_cut, InputCut):
                    # The attribution must be the watched tensor given from 
                    # `qoi_bprop`.
                    attribution_results = intervention

                else:
                    attribution_layers = (
                        self._get_logit_layer() if
                        (isinstance(attribution_cut,
                                    LogitCut)) else self._get_output_layer() if
                        (isinstance(attribution_cut, OutputCut)) else
                        self._get_layers_by_name(attribution_cut.name))

                    attribution_results = [None for _ in attribution_layers]

                    for i, layer in enumerate(attribution_layers):
                        if self._is_input_layer(layer):
                            # Input layers don't end up calling the hook, so we
                            # have to get their output manually.
                            attribution_results[i] = intervention[
                                self._input_layer_index(layer)]

                        else:
                            layer.retrieve_functions.append(
                                retrieve_index(
                                    i, attribution_results,
                                    attribution_cut.anchor))

            # Run a point.
            self._model(*model_args, **model_kwargs)

        finally:
            # Clear the hooks after running the model so that `fprop` doesn't
            # leave the model in an altered state.
            self._clear_hooks()

        if return_numpy:
            results = ModelWrapper._nested_apply(
                results, lambda t: t.numpy()
                if not isinstance(t, np.ndarray) else t)

        return (results, attribution_results) if attribution_cut else results

    def qoi_bprop(
            self,
            qoi,
            model_args,
            model_kwargs={},
            doi_cut=None,
            to_cut=None,
            attribution_cut=None,
            intervention=None):
        """
        qoi_bprop Run the model from the from_layer to the qoi layer
            and give the gradients w.r.t `attribution_cut`

        Parameters
        ----------
        model_args, model_kwargs: 
            The args and kwargs given to the call method of a model.
            This should represent the instances to obtain attributions for, 
            assumed to be a *batched* input. if `self.model` supports evaluation 
            on *data tensors*, the  appropriate tensor type may be used (e.g., 
            Pytorch models may accept Pytorch tensors in addition to 
            `np.ndarray`s). The shape of the inputs must match the input shape
            of `self.model`.
        
        qoi: a Quantity of Interest
            This method will accumulate all gradients of the qoi w.r.t 
            `attribution_cut`
        doi_cut: Cut, 
            if `doi_cut` is None, this refers to the InputCut. Cut from which to
            begin propagation. The shape of `intervention` must match the output
            shape of this layer.
        attribution_cut: Cut, optional
            if `attribution_cut` is None, this refers to the InputCut.
            The Cut in which attribution will be calculated. This is generally
            taken from the attribution slyce's attribution_cut.
        to_cut: Cut, optional
            if `to_cut` is None, this refers to the OutputCut.
            The Cut in which qoi will be calculated. This is generally
            taken from the attribution slyce's to_cut.
        intervention : backend.Tensor or np.array
            Input tensor to propagate through the model. If an np.array,
            will be converted to a tensor on the same device as the model.

        Returns
        -------
        (backend.Tensor or np.ndarray)
            the gradients of `qoi` w.r.t. `attribution_cut`, keeping same type 
            as the input.
        """
        if intervention is None:
            intervention = model_args

        if not self._eager:
            return super().qoi_bprop(
                model_args, model_kwargs, doi_cut, to_cut, attribution_cut,
                intervention)

        if attribution_cut is None:
            attribution_cut = InputCut()
        if to_cut is None:
            to_cut = OutputCut()

        return_numpy = True

        with tf.GradientTape(persistent=True) as tape:

            intervention = intervention if isinstance(
                intervention, DATA_CONTAINER_TYPE) else [intervention]
            # We return a numpy array if we were given a numpy array; otherwise
            # we will let the returned values remain data tensors.
            return_numpy = isinstance(intervention, np.ndarray) or isinstance(
                intervention[0], np.ndarray)

            # Convert `intervention` to a data tensor if it isn't already.

            if return_numpy:
                intervention = [
                    ModelWrapper._nested_apply(x_i, tf.constant)
                    for x_i in intervention
                ]

            for x_i in intervention:
                ModelWrapper._nested_apply(x_i, tape.watch)

            outputs, attribution_features = self.fprop(
                model_args,
                model_kwargs,
                doi_cut=doi_cut if doi_cut else InputCut(),
                to_cut=to_cut,
                attribution_cut=attribution_cut,
                intervention=intervention)
            if isinstance(outputs, DATA_CONTAINER_TYPE) and isinstance(
                    outputs[0], DATA_CONTAINER_TYPE):
                outputs = outputs[0]

            Q = qoi(outputs[0]) if len(outputs) == 1 else qoi(outputs)
            if isinstance(Q, DATA_CONTAINER_TYPE) and len(Q) == 1:
                Q = B.sum(Q)

        grads = [
            tape.gradient(q, attribution_features) for q in Q
        ] if isinstance(Q, DATA_CONTAINER_TYPE) else tape.gradient(
            Q, attribution_features)

        grads = grads[0] if isinstance(
            grads, DATA_CONTAINER_TYPE) and len(grads) == 1 else grads

        grads = [attribution_cut.access_layer(g) for g in grads] if isinstance(
            grads,
            DATA_CONTAINER_TYPE) else attribution_cut.access_layer(grads)

        del tape

        if return_numpy:
            grads = [
                ModelWrapper._nested_apply(g, B.as_array) for g in grads
            ] if isinstance(
                grads, DATA_CONTAINER_TYPE) else ModelWrapper._nested_apply(
                    grads, B.as_array)

        return grads[0] if isinstance(
            grads, DATA_CONTAINER_TYPE) and len(grads) == 1 else grads
