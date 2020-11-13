# TODO(klas): this code is stale and needs to be re-written with the changes to
#   the API, e.g., Cuts.

import numpy as np
import tensorflow as tf

from netlens import backend as B
from netlens.slices import InputCut, OutputCut, LogitCut
from netlens.models._model_base import ModelWrapper


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
            default_feed_dict=None,
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
        default_feed_dict : dict, optional
            Default feed_dict to be passed to `session.run()`. This is useful 
            for defining PlaceHolders that do not change per model run.
        """

        self._graph = graph

        self._inputs = (
            input_tensors
            if isinstance(input_tensors, list) else [input_tensors])
        self._outputs = (
            output_tensors
            if isinstance(output_tensors, list) else [output_tensors])
        self._internal_tensors = (
            internal_tensor_dict if internal_tensor_dict is not None else {})

        self._default_feed_dict = (
            default_feed_dict if default_feed_dict is not None else {})

        self._session = session

        # Bprop may need some feed dict values that were set during the fprop stage.
        # This cache will naively save the last fprop feed dict.
        self._cached_fprop_feed_dict = {}

        # This cache will be used to not recreate gradient nodes if they have already been created.
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

        elif isinstance(cut.name, list):
            layers = [self._get_layer(name) for name in cut.name]

        else:
            layers = self._get_layer(cut.name)

        return layers if isinstance(layers, list) else [layers]

    def fprop(self, x, from_cut=None, to_cut=None):
        """
        fprop Forward propagate the model

        Parameters
        ----------
        x: 
            Input tensors to propagate through the model. If a list, they must match the ordering of from_cut.
            x can also be a feed_dict, in which case from_cut will be the feed dict keys 
        qoi: a Quantity of Interest
            This method will accumulate all gradients of the qoi w.r.t x
        from_cut: Cut, optional
            The Cut from which to begin propagation. The shape of `x` must
            match the input shape of this layer. By default 0.
        to_cut : Cut, optional
            The Cut to return output activation tensors for. If None,
            assumed to be just the final layer. By default None

        Returns
        -------
        (list of backend.Tensor)
            A list of output activations are returned.
        """
        if from_cut is None:
            from_cut = InputCut()
        if to_cut is None:
            to_cut = OutputCut()

        from_tensors = self._get_layers(from_cut)
        to_tensors = self._get_layers(to_cut)

        if isinstance(x, dict):
            for key_tensor in x:
                assert tf.is_tensor(key_tensor)
            from_tensors = x.keys()
            x = [x[key_tensor] for key_tensor in from_tensors]

        # Convert `x` to a list of inputs if it isn't already.
        if not isinstance(x, list):
            x = [x]

        # Tensorlow doesn't allow you to make a function that returns the same
        # tensor as it takes in. Thus, we have to have a special case for the
        # identity function. Any tensors that are both in `from_tensors` and
        # `to_tensors` cannot be computed via a `keras.backend.function` and
        # thus need to be taken from the input, `x`.
        identity_map = {
            i: j for i, to_tensor in enumerate(to_tensors)
            for j, from_tensor in enumerate(from_tensors)
            if to_tensor == from_tensor
        }

        non_identity_to_tensors = [
            to_tensor for i, to_tensor in enumerate(to_tensors)
            if i not in identity_map
        ]

        # Compute the output values of `to_tensors` unless all `to_tensor`s were
        # also `from_tensors`.
        if non_identity_to_tensors:

            feed_dict = dict(self._default_feed_dict)
            feed_dict.update(
                {from_tensor: xi for from_tensor, xi in zip(from_tensors, x)})

            self._cached_fprop_feed_dict = feed_dict

            out_vals = self._run_session(non_identity_to_tensors, feed_dict)

        else:
            out_vals = []

        # For any `to_tensor`s that were also `from_tensor`s, insert the
        # corresponding concrete input value from `x` in the output's place.
        for i in sorted(identity_map):
            out_vals.insert(i, x[identity_map[i]])

        return out_vals

    def _run_session(self, outs, feed_dict):
        if self._session is not None:
            return self._session.run(outs, feed_dict=feed_dict)

        else:
            with tf.Session(graph=self._graph) as session:
                session.run(tf.global_variables_initializer())
                return session.run(outs, feed_dict=feed_dict)

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
        np.Array or list of np.Array
            the gradients of `qoi` w.r.t. `from_cut`
        """
        if from_cut is None:
            from_cut = InputCut()
        if to_cut is None:
            to_cut = OutputCut()

        input_cut = doi_cut if doi_cut else InputCut()

        latent_tensors = self._get_layers(from_cut)
        to_tensors = self._get_layers(to_cut)
        input_tensors = self._get_layers(input_cut)

        # Convert `x` to a list of inputs if it isn't already.
        if not isinstance(x, list):
            x = [x]

        feed_dict = dict(self._default_feed_dict)
        feed_dict.update(self._cached_fprop_feed_dict)
        feed_dict.update(
            {input_tensor: xi for input_tensor, xi in zip(input_tensors, x)})

        doi_repeated_batch_size = x[0].shape[0]
        for k in feed_dict:
            val = feed_dict[k]
            if isinstance(val, np.ndarray):
                doi_resolution = int(doi_repeated_batch_size / val.shape[0])
                tile_shape = [1] * len(val.shape)
                tile_shape[0] = doi_resolution
                feed_dict[k] = np.tile(val, tuple(tile_shape))

        z_grads = []
        with self._graph.as_default():
            for z in latent_tensors:
                gradient_tensor_key = (z, frozenset(to_tensors))
                if gradient_tensor_key in self._cached_gradient_tensors:
                    grads = self._cached_gradient_tensors[gradient_tensor_key]
                else:
                    Q = qoi(to_tensors[0]) if len(to_tensors) == 1 else qoi(
                        to_tensors)

                    grads = [B.gradient(q, z)[0] for q in Q] if isinstance(
                        Q, list) else B.gradient(Q, z)[0]
                    grads = grads[0] if isinstance(
                        grads, list) and len(grads) == 1 else grads
                    grads = [from_cut.access_layer(g) for g in grads
                            ] if isinstance(
                                grads, list) else from_cut.access_layer(grads)
                    self._cached_gradient_tensors[gradient_tensor_key] = grads
                z_grads.append(grads)

        grad_flat = ModelWrapper._flatten(z_grads)
        gradients = [self._run_session(g, feed_dict) for g in grad_flat]

        gradients = ModelWrapper._unflatten(gradients, z_grads)
        return gradients[0] if len(gradients) == 1 else gradients
