# TODO(klas): this code is stale and needs to be re-written with the changes to
#   the API, e.g., Cuts.

import numpy as np
import tensorflow as tf

from trulens.nn import backend as B
from trulens.nn.slices import InputCut, OutputCut, LogitCut
from trulens.nn.models._model_base import ModelWrapper, DATA_CONTAINER_TYPE


class TensorflowModelWrapper(ModelWrapper):
    """
    Model wrapper that exposes the internal components
    of Tensorflow objects.
    """

    def __init__(
            self,
            graph,
            input_tensors,
            output_tensors,
            internal_tensor_dict=None,
            session=None):
        """
        Parameters
        ----------
        graph : tf.Graph
            The computation graph representing the model.
        input_tensors : B.Tensor | list of B.Tensor
            A list of the tensors that are the inputs to the graph. If there is
            only one input, it can be given without wrapping it in a list.
            This is needed as the input tensors of a graph cannot be inferred.
        output_tensors : B.Tensor | list of B.Tensor
            A list of the tensors that are the outputs to the graph. If there is
            only one output, it can be given without wrapping it in a list.
            This is needed as the output tensors of a graph cannot be inferred.
        internal_tensor_dict : dict of str -> tensor, optional
            The user can specify their own accessors to the tensors they would
            like to expose in the graph by passing a dict that maps their chosen
            name to the corresponding tensor. Any tensors not given in
            `internal_tensor_dict` can be accessed via the name given to them by
            tensorflow.
        """

        self._graph = graph

        self._inputs = (
            input_tensors if isinstance(input_tensors, DATA_CONTAINER_TYPE) else
            [input_tensors])
        self._outputs = (
            output_tensors if isinstance(output_tensors, DATA_CONTAINER_TYPE)
            else [output_tensors])
        self._internal_tensors = (
            internal_tensor_dict if internal_tensor_dict is not None else {})

        self._session = session

        # This cache will be used to not recreate gradient nodes if they have
        # already been created.
        self._cached_gradient_tensors = {}

    def _get_layer(self, name):
        if name in self._internal_tensors:
            return self._internal_tensors[name]
        else:
            try:
                return self._graph.get_tensor_by_name(name)

            except KeyError:
                raise ValueError('No such layer tensor:', name)

    def _get_layers(self, cut):
        if isinstance(cut, InputCut):
            return self._inputs

        elif isinstance(cut, OutputCut):
            return self._outputs

        elif isinstance(cut, LogitCut):
            if 'logits' in self._internal_tensors:
                layers = self._internal_tensors['logits']

            else:
                raise ValueError(
                    '`LogitCut` was used, but the model has not specified the '
                    'tensors that correspond to the logit output.')

        elif isinstance(cut.name, DATA_CONTAINER_TYPE):
            layers = [self._get_layer(name) for name in cut.name]

        else:
            layers = self._get_layer(cut.name)

        return layers if isinstance(layers, DATA_CONTAINER_TYPE) else [layers]

    def _prepare_feed_dict_with_intervention(
            self, model_args, model_kwargs, intervention, doi_tensors):

        feed_dict = {}
        input_tensors = model_args
        if isinstance(
                input_tensors,
                DATA_CONTAINER_TYPE) and len(input_tensors) > 0 and isinstance(
                    input_tensors[0], DATA_CONTAINER_TYPE):
            input_tensors = input_tensors[0]

        # If inputs are supplied via args.
        if (len(input_tensors) == len(self._inputs) and
                len(input_tensors) > 0 and
            (tf.is_tensor(input_tensors[0]) or isinstance(input_tensors[0],
                                                          (np.ndarray)))):
            feed_dict.update(
                {
                    input_tensor: xi
                    for input_tensor, xi in zip(self._inputs, input_tensors)
                })

        # If inputs are supplied via feed dict
        if ('feed_dict' in model_kwargs):
            feed_dict.update(model_kwargs['feed_dict'])

        # Convert `intervention` to a list of inputs if it isn't already.
        if intervention is not None:
            if isinstance(intervention, dict):
                for key_tensor in intervention:
                    assert tf.is_tensor(
                        key_tensor
                    ), "Obtained a non tensor in feed dict: {}".format(
                        str(key_tensor))
                doi_tensors = intervention.keys()
                intervention = [
                    intervention[key_tensor] for key_tensor in doi_tensors
                ]

            if not isinstance(intervention, DATA_CONTAINER_TYPE):
                intervention = [intervention]

            feed_dict.update(
                {
                    input_tensor: xi
                    for input_tensor, xi in zip(doi_tensors, intervention)
                })

            doi_repeated_batch_size = intervention[0].shape[0]
            for k in feed_dict:
                val = feed_dict[k]
                if isinstance(val, np.ndarray):
                    doi_resolution = int(doi_repeated_batch_size / val.shape[0])
                    tile_shape = [1] * len(val.shape)
                    tile_shape[0] = doi_resolution
                    feed_dict[k] = np.tile(val, tuple(tile_shape))

        elif intervention is None and doi_tensors == self._inputs:
            intervention = [feed_dict[key_tensor] for key_tensor in doi_tensors]

        else:
            intervention = []

        return feed_dict, intervention

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
            apply distributions of interest (DoI).
        to_cut : Cut, optional
            The Cut to return output activation tensors for. If `None`,
            assumed to be just the final layer. By default None
        attribution_cut : Cut, optional
            An Cut to return activation tensors for. If `None` 
            attributions layer output is not returned.
        intervention : backend.Tensor or np.array
            Input tensor to propagate through the model. If an np.array, will be
            converted to a tensor on the same device as the model. Intervention
            can also be a `feed_dict`.

        Returns
        -------
        (list of backend.Tensor or np.ndarray)
            A list of output activations are returned, keeping same type as the
            input. If `attribution_cut` is supplied, also return the cut 
            activations.
        """

        if doi_cut is None:
            doi_cut = InputCut()
        if to_cut is None:
            to_cut = OutputCut()

        doi_tensors = self._get_layers(doi_cut)
        to_tensors = self._get_layers(to_cut)

        feed_dict, intervention = self._prepare_feed_dict_with_intervention(
            model_args, model_kwargs, intervention, doi_tensors)

        # Tensorlow doesn't allow you to make a function that returns the same
        # tensor as it takes in. Thus, we have to have a special case for the
        # identity function. Any tensors that are both in `doi_tensors` and
        # `to_tensors` cannot be computed via a `keras.backend.function` and
        # thus need to be taken from the input, `x`.
        identity_map = {
            i: j for i, to_tensor in enumerate(to_tensors)
            for j, from_tensor in enumerate(doi_tensors)
            if to_tensor == from_tensor
        }

        non_identity_to_tensors = [
            to_tensor for i, to_tensor in enumerate(to_tensors)
            if i not in identity_map
        ]

        # Compute the output values of `to_tensors` unless all `to_tensor`s were
        # also `doi_tensors`.
        if non_identity_to_tensors:
            out_vals = self._run_session(non_identity_to_tensors, feed_dict)

        else:
            out_vals = []

        # For any `to_tensor`s that were also `from_tensor`s, insert the
        # corresponding concrete input value from `x` in the output's place.
        for i in sorted(identity_map):
            out_vals.insert(i, intervention[identity_map[i]])

        return out_vals

    def _run_session(self, outs, feed_dict):
        if self._session is not None:
            return self._session.run(outs, feed_dict=feed_dict)

        else:
            with tf.Session(graph=self._graph) as session:
                session.run(tf.global_variables_initializer())
                return session.run(outs, feed_dict=feed_dict)

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
            `attribution_cut`.
        doi_cut: Cut, 
            if `doi_cut` is None, this refers to the InputCut.
            Cut from which to begin propagation. The shape of `intervention`
            must match the output shape of this layer.
        attribution_cut: Cut, optional
            if `attribution_cut` is None, this refers to the InputCut.
            The Cut in which attribution will be calculated. This is generally
            taken from the attribution slyce's attribution_cut.
        to_cut: Cut, optional
            if `to_cut` is None, this refers to the OutputCut.
            The Cut in which qoi will be calculated. This is generally
            taken from the attribution slyce's to_cut.
        intervention : backend.Tensor or np.array
            Input tensor to propagate through the model. If an np.array, will be
            converted to a tensor on the same device as the model.
            intervention can also be a feed_dict

        Returns
        -------
        (backend.Tensor or np.ndarray)
            the gradients of `qoi` w.r.t. `attribution_cut`, keeping same type 
            as the input.
        """
        if attribution_cut is None:
            attribution_cut = InputCut()
        if to_cut is None:
            to_cut = OutputCut()

        doi_cut = doi_cut if doi_cut else InputCut()

        attribution_tensors = self._get_layers(attribution_cut)
        to_tensors = self._get_layers(to_cut)
        doi_tensors = self._get_layers(doi_cut)

        feed_dict, _ = self._prepare_feed_dict_with_intervention(
            model_args, model_kwargs, intervention, doi_tensors)

        z_grads = []
        with self._graph.as_default():
            for z in attribution_tensors:
                gradient_tensor_key = (z, frozenset(to_tensors))
                if gradient_tensor_key in self._cached_gradient_tensors:
                    grads = self._cached_gradient_tensors[gradient_tensor_key]
                else:
                    Q = qoi(to_tensors[0]) if len(to_tensors) == 1 else qoi(
                        to_tensors)

                    grads = [B.gradient(q, z)[0] for q in Q] if isinstance(
                        Q, DATA_CONTAINER_TYPE) else B.gradient(Q, z)[0]
                    grads = grads[0] if isinstance(
                        grads,
                        DATA_CONTAINER_TYPE) and len(grads) == 1 else grads
                    grads = [
                        attribution_cut.access_layer(g) for g in grads
                    ] if isinstance(
                        grads, 
                        DATA_CONTAINER_TYPE) else attribution_cut.access_layer(
                            grads)
                    self._cached_gradient_tensors[gradient_tensor_key] = grads
                z_grads.append(grads)

        grad_flat = ModelWrapper._flatten(z_grads)
        gradients = [self._run_session(g, feed_dict) for g in grad_flat]

        gradients = ModelWrapper._unflatten(gradients, z_grads)
        return gradients[0] if len(gradients) == 1 else gradients
