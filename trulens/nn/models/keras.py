import numpy as np
import os
import tempfile
import importlib

from trulens.utils import tru_logger

from trulens.nn.backend import get_backend, Backend
from trulens.nn.slices import InputCut, OutputCut, LogitCut
from trulens.nn.models._model_base import ModelWrapper, DATA_CONTAINER_TYPE


def import_keras_backend():
    '''
    Dynamically import keras in case backend changes dynamically
    '''
    if get_backend().backend == Backend.KERAS:
        return importlib.import_module(name='keras')
    elif get_backend().backend == Backend.TF_KERAS or get_backend(
    ).backend == Backend.TENSORFLOW:
        return importlib.import_module(name='tensorflow.keras')


class KerasModelWrapper(ModelWrapper):
    """
    Model wrapper that exposes internal layers of Keras models.
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
        model : keras.Model
            Keras model instance to wrap
        custom_objects : list of keras.Layer, optional
            If the model uses any user-defined layers, they must be passed
            in as a list. By default `None`.
        replace_softmax : bool, optional
            If `True`, then remove the final layer's activation function.
            By default `False`.
        clone_model : bool, optional
            Whether to make a copy of the target model. If this is `False` and
            replace_softmax is `True`, then the passed-in model will be
            modified. By default `False`.
        """
        self.keras = import_keras_backend()
        if not isinstance(model, self.keras.models.Model):
            raise ValueError(
                'Model must be an instance of `{}.models.Model`.\n\n'
                '(you may be seeing this error if you passed a '
                '`tensorflow.keras` model while using the \'keras\' backend or '
                'vice-versa)'.format(get_backend().backend))

        if replace_softmax:
            self._model = KerasModelWrapper._replace_probits_with_logits(
                model,
                probits_layer_name=softmax_layer,
                custom_objects=custom_objects)

            self._logit_layer = softmax_layer

        else:
            self._model = model
            self._logit_layer = logit_layer

        self._layers = model.layers
        self._layernames = [l.name for l in self._layers]

    @staticmethod
    def _replace_probits_with_logits(
            model, probits_layer_name=-1, custom_objects=None):
        """
        _replace_softmax_with_logits Remove the final layer's activation 
        function

        When computing gradient-based attribution methods, better results 
        usually obtain by removing the typical softmax activation function.

        Parameters
        ----------
        model : keras.Model
            Target model
        softmax_layer : int, optional
            Layer containing relevant activation, by default -1
        custom_objects : list of keras.Layer, optional
            If the model uses any user-defined layers, they must be passed in. 
            By default None.

        Returns
        -------
        keras.Model
            Clone of original model with specified activation function removed.
        """
        probits_layer = (
            model.get_layer(probits_layer_name) if isinstance(
                probits_layer_name, str) else model.layers[probits_layer_name])

        # Make sure the specified layer has an activation.
        try:
            activation = probits_layer.activation
        except AttributeError:
            raise ValueError(
                'The specified layer to `_replace_probits_with_logits` has no '
                '`activation` attribute. The specified layer should convert '
                'its input to probits, i.e., it should have an `activation` '
                'attribute that is either a softmax or a sigmoid.')

        # Warn if the specified layer's activation is not a softmax or a
        # sigmoid.
        if not (activation == self.keras.activations.softmax or
                activation == self.keras.activations.sigmoid):
            tru_logger.warn(
                'The activation of the specified layer to '
                '`_replace_probits_with_logits` is not a softmax or a sigmoid; '
                'it may not currently convert its input to probits.')

        try:
            # Temporarily replace the softmax activation.
            probits_layer.activation = self.keras.activations.linear

            # Save and reload the model; this ensures we get a clone of the
            # model, but with the activation updated.
            tmp_path = os.path.join(
                tempfile.gettempdir(),
                next(tempfile._get_candidate_names()) + '.h5')

            model.save(tmp_path)

            return self.keras.models.load_model(
                tmp_path, custom_objects=custom_objects)

        finally:
            # Revert the activation in the original model back to its original
            # state.
            probits_layer.activation = activation

            # Remove the temporary file.
            os.remove(tmp_path)

    def _get_logit_layer(self):
        if self._logit_layer is not None:
            return self._get_layers_by_name(self._logit_layer)

        elif 'logits' in self._layernames:
            return self._get_layers_by_name('logits')

        else:
            raise ValueError(
                '`LogitCut` was used, but the model has not specified the '
                'layer whose outputs correspond to the logit output.\n\n'
                'To use the `LogitCut`, either ensure that the model has a '
                'layer named "logits", specify the `logit_layer` in the '
                'model wrapper constructor, or use the `replace_softmax` '
                'option in the model wrapper constructor.')

    def _get_layers_by_name(self, name):
        '''
        _get_layers_by_name Return a list of layers specified by the `name`
        field in a cut.

        Parameters
        ----------
        name : int | str | list of int or str

        Returns
        -------
        list of keras.backend.Layer

        Raises
        ------
        ValueError
            Unsupported type for cut name.
        ValueError
            No layer with given name string identifier.
        ValueError
            Layer index out of bounds.
        '''
        if isinstance(name, str):
            if not name in self._layernames:
                raise ValueError('No such layer tensor:', name)

            return [self._model.get_layer(name=name)]

        elif isinstance(name, int):
            if len(self._layers) <= name:
                raise ValueError('Layer index out of bounds:', name)

            return [self._model.get_layer(index=name)]

        elif isinstance(name, DATA_CONTAINER_TYPE):
            return sum([self._get_layers_by_name(n) for n in name], [])

        else:
            raise ValueError('Unsupported type for cut name:', type(name))

    def _get_layers(self, cut):
        '''
        get_layer Return the tensor(s) representing the layer(s) specified by 
        the given cut.

        Parameters
        ----------
        cut : Cut
            Cut specifying which tensor in the model to return.

        Returns
        -------
        list of keras.backend.Tensor
            Tensors representing the layer(s) specified by the given cut.

        Raises
        ------
        ValueError
            Unsupported type for cut name.
        ValueError
            No layer with given name string identifier.
        ValueError
            Layer index out of bounds.
        '''
        if isinstance(cut, InputCut):
            return self._model.inputs

        elif isinstance(cut, OutputCut):
            return self._model.outputs

        elif isinstance(cut, LogitCut):
            layers = self._get_logit_layer()

        else:
            layers = self._get_layers_by_name(cut.name)

        flat = lambda l: [
            item for items in l for item in
            (items if isinstance(items, DATA_CONTAINER_TYPE) else [items])
        ]

        if cut.anchor not in ['in', 'out']:
            return flat(
                [
                    layer.output[cut.anchor][0]
                    if cut.anchor in layer.output else layer.output
                    for layer in layers
                ])
        return (
            flat([layer.input for layer in layers])
            if cut.anchor == 'in' else flat([layer.output for layer in layers]))

    def _prepare_intervention_with_input(
            self, model_args, intervention, doi_tensors):
        input_tensors = self._get_layers(InputCut())
        if not all(elem in doi_tensors for elem in input_tensors):
            intervention_batch_doi_len = len(intervention[0])
            model_args_batch_len = len(model_args[0])
            doi_tensors.extend(input_tensors)
            if intervention_batch_doi_len != model_args_batch_len:
                doi_factor = intervention_batch_doi_len / model_args_batch_len
                model_args_expanded = [
                    get_backend().stack([arg] * int(doi_factor))
                    for arg in model_args
                ]
                intervention.extend(model_args_expanded)
            else:
                intervention.extend(model_args)
        return doi_tensors, intervention

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
            Pytorch models may accept Pytorch tensors in additon to 
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
            An Cut to return activation tensors for. If `None`, 
            assumed to be the doi_cut
            
        intervention : backend.Tensor or np.array
            Input tensor to propagate through the model. If an np.array, 
            will be converted to a tensor on the same device as the model.

        Returns
        -------
        (list of backend.Tensor or np.ndarray)
            A list of output activations are returned, keeping the same type as
            the input. If `attribution_cut` is supplied, also return the cut 
            activations.
        """
        if doi_cut is None:
            doi_cut = InputCut()
        if to_cut is None:
            to_cut = OutputCut()

        doi_tensors = self._get_layers(doi_cut)
        to_tensors = self._get_layers(to_cut)

        if intervention is None:
            intervention = model_args[0]

        # Convert `intervention` to a list of inputs if it isn't already.
        if not isinstance(intervention, DATA_CONTAINER_TYPE):
            intervention = [intervention]

        # Tensorflow doesn't allow you to make a function that returns the same
        # tensor as it takes in. Thus, we have to have a special case for the
        # identity function. Any tensors that are both in `doi_tensors` and
        # `to_tensors` cannot be computed via a `keras.backend.function` and
        # thus need to be taken from the input, `intervention`.
        identity_map = {
            i: j for i, to_tensor in enumerate(to_tensors)
            for j, from_tensor in enumerate(doi_tensors)
            if to_tensor is from_tensor
        }

        non_identity_to_tensors = [
            to_tensor for i, to_tensor in enumerate(to_tensors)
            if i not in identity_map.keys()
        ]
        # Compute the output values of `to_tensors` unless all `to_tensor`s were
        # also `doi_tensors`.
        if non_identity_to_tensors:
            fprop_fn = self.keras.backend.function(
                doi_tensors, non_identity_to_tensors)
            out_vals = fprop_fn(intervention)
            del fprop_fn

        else:
            out_vals = []

        # For any `to_tensor`s that were also `from_tensor`s, insert the
        # corresponding concrete input value from `intervention` in the output's
        # place.
        for i in sorted(identity_map):
            out_vals.insert(i, intervention[identity_map[i]])

        return out_vals

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
            Pytorch models may accept Pytorch tensors in additon to
            `np.ndarray`s). The shape of the inputs must match the input shape
            of `self.model`. 
        
        qoi: a Quantity of Interest
            This method will accumulate all gradients of the qoi w.r.t
            `attribution_cut`
        doi_cut: Cut, 
            If `doi_cut` is None, this refers to the InputCut. Cut from which to
            begin propagation. The shape of `intervention` must match the output
            shape of this layer.
        attribution_cut: Cut, optional
            If `attribution_cut` is None, this refers to the InputCut. The Cut
            in which attribution will be calculated. This is generally taken
            from the attribution slyce's attribution_cut.
        to_cut: Cut, optional
            If `to_cut` is None, this refers to the OutputCut. The Cut in which
            qoi will be calculated. This is generally taken from the attribution
            slice's `to_cut`.
        intervention : backend.Tensor or np.array
            Input tensor to propagate through the model. If an np.array,
            will be converted to a tensor on the same device as the model.

        Returns
        -------
        (backend.Tensor or np.ndarray)
            The gradients of `qoi` w.r.t. `attribution_cut`, keeping same type
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
        if intervention is None:
            intervention = model_args

        intervention = intervention if isinstance(
            intervention, DATA_CONTAINER_TYPE) else [intervention]

        Q = qoi(to_tensors[0]) if len(to_tensors) == 1 else qoi(to_tensors)

        doi_tensors, intervention = self._prepare_intervention_with_input(
            model_args, intervention, doi_tensors)

        gradients = [
            self.keras.backend.function(
                doi_tensors,
                get_backend().gradient(q, attribution_tensors))(intervention)
            for q in Q
        ] if isinstance(
            Q, DATA_CONTAINER_TYPE) else self.keras.backend.function(
                doi_tensors,
                get_backend().gradient(Q, attribution_tensors))(intervention)

        return gradients[0] if len(gradients) == 1 else gradients
