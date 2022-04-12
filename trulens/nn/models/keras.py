import importlib
import os
import tempfile
from typing import Tuple

from trulens.nn.backend import Backend
from trulens.nn.backend import get_backend
from trulens.nn.models._model_base import ModelWrapper
from trulens.nn.quantities import QoI
from trulens.nn.slices import Cut
from trulens.nn.slices import InputCut
from trulens.nn.slices import LogitCut
from trulens.nn.slices import OutputCut
from trulens.utils import tru_logger
from trulens.utils.typing import DATA_CONTAINER_TYPE
from trulens.utils.typing import ModelInputs
from trulens.utils.typing import om_of_many
from trulens.utils.typing import Outputs
from trulens.utils.typing import TensorArgs
from trulens.utils.typing import TensorLike


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
        custom_objects=None
    ):
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
                'vice-versa)'.format(get_backend().backend)
            )

        if replace_softmax:
            self._model = KerasModelWrapper._replace_probits_with_logits(
                model,
                probits_layer_name=softmax_layer,
                custom_objects=custom_objects
            )

            self._logit_layer = softmax_layer

        else:
            self._model = model
            self._logit_layer = logit_layer

        self._layers = model.layers
        self._layernames = [l.name for l in self._layers]

    @staticmethod
    def _replace_probits_with_logits(
        model, probits_layer_name=-1, custom_objects=None
    ):
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
            model.get_layer(probits_layer_name)
            if isinstance(probits_layer_name, str) else
            model.layers[probits_layer_name]
        )

        # Make sure the specified layer has an activation.
        try:
            activation = probits_layer.activation
        except AttributeError:
            raise ValueError(
                'The specified layer to `_replace_probits_with_logits` has no '
                '`activation` attribute. The specified layer should convert '
                'its input to probits, i.e., it should have an `activation` '
                'attribute that is either a softmax or a sigmoid.'
            )

        # Warn if the specified layer's activation is not a softmax or a
        # sigmoid.
        if not (activation == self.keras.activations.softmax or
                activation == self.keras.activations.sigmoid):
            tru_logger.warn(
                'The activation of the specified layer to '
                '`_replace_probits_with_logits` is not a softmax or a sigmoid; '
                'it may not currently convert its input to probits.'
            )

        try:
            # Temporarily replace the softmax activation.
            probits_layer.activation = self.keras.activations.linear

            # Save and reload the model; this ensures we get a clone of the
            # model, but with the activation updated.
            tmp_path = os.path.join(
                tempfile.gettempdir(),
                next(tempfile._get_candidate_names()) + '.h5'
            )

            model.save(tmp_path)

            return self.keras.models.load_model(
                tmp_path, custom_objects=custom_objects
            )

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
                'option in the model wrapper constructor.'
            )

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
                ]
            )
        return (
            flat([layer.input for layer in layers])
            if cut.anchor == 'in' else flat([layer.output for layer in layers])
        )

    def _prepare_intervention_with_input(
        self, model_args, intervention, doi_tensors
    ):
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

    def _fprop(
        self, *, model_inputs: ModelInputs, doi_cut: Cut, to_cut: Cut,
        attribution_cut: Cut, intervention: TensorArgs
    ) -> Tuple[Outputs[TensorLike], Outputs[TensorLike]]:
        """
        See ModelWrapper.fprop .
        """

        B = get_backend()

        # TODO: use the ModelInputs structure in this backend
        intervention = intervention.args
        model_args = model_inputs.args
        model_kwargs = model_inputs.kwargs

        doi_tensors = self._get_layers(doi_cut)
        to_tensors = self._get_layers(to_cut)

        def _tensor(k):
            if isinstance(k, B.Tensor):
                return k
            # any named inputs must correspond to layer names
            return self._model.get_layer(k).output

        val_map = dict()

        # construct a feed_dict

        # Model inputs come from model_args
        val_map.update(
            {
                k: v for k, v in
                zip(self._model.inputs[0:len(model_args)], model_args)
            }
        )
        # Other placeholders come from kwargs.
        val_map.update({_tensor(k): v for k, v in model_kwargs.items()})

        # Finally, interventions override any previously set tensors.
        val_map.update({k: v for k, v in zip(doi_tensors, intervention)})

        all_inputs = list(val_map.keys())
        all_vals = list(val_map.values())

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
                inputs=all_inputs, outputs=non_identity_to_tensors
            )
            out_vals = fprop_fn(all_vals)
            del fprop_fn

        else:
            out_vals = []

        # For any `to_tensor`s that were also `from_tensor`s, insert the
        # corresponding concrete input value from `intervention` in the output's
        # place.

        for i in sorted(identity_map):
            out_vals.insert(i, intervention[identity_map[i]])

        # internal _fprop returns two things in general
        return (out_vals, None)

    def _qoi_bprop(
        self, *, qoi: QoI, model_inputs: ModelInputs, doi_cut: Cut, to_cut: Cut,
        attribution_cut: Cut, intervention: TensorArgs
    ):
        """
        See ModelWrapper.qoi_bprop .
        """

        attribution_tensors = self._get_layers(attribution_cut)
        to_tensors = self._get_layers(to_cut)
        doi_tensors = self._get_layers(doi_cut)

        to_tensors = om_of_many(to_tensors)
        Q = qoi._wrap_public_call(to_tensors)

        int_copy = list(x for x in intervention.args)
        doi_copy = list(x for x in doi_tensors)

        doi_tensors, intervention_args = self._prepare_intervention_with_input(
            model_inputs.args, int_copy, doi_copy
        )

        gradients = [
            self.keras.backend.function(
                doi_tensors,
                get_backend().gradient(q, attribution_tensors)
            )(intervention_args) for q in Q
        ]

        return gradients
