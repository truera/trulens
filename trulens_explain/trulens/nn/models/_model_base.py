""" 
The TruLens library is designed to support models implemented via a variety of
different popular python neural network frameworks: Keras (with TensorFlow or 
Theano backend), TensorFlow, and Pytorch. In order provide the same 
functionality to models made with frameworks that implement things (e.g., 
gradient computations) a number of different ways, we provide an adapter class 
to provide a unified model API. In order to compute attributions for a model, 
it should be wrapped as a `ModelWrapper` instance.
"""
from abc import ABC as AbstractBaseClass
from abc import abstractmethod
from typing import List, Optional, Tuple, Type, Union

import numpy as np
from trulens.nn.backend import get_backend
from trulens.nn.quantities import QoI
from trulens.nn.slices import Cut
from trulens.nn.slices import InputCut
from trulens.nn.slices import OutputCut
from trulens.utils import tru_logger
from trulens.utils.typing import ArgsLike
from trulens.utils.typing import DATA_CONTAINER_TYPE
from trulens.utils.typing import Inputs
from trulens.utils.typing import InterventionLike
from trulens.utils.typing import KwargsLike
from trulens.utils.typing import many_of_om
from trulens.utils.typing import ModelInputs
from trulens.utils.typing import nested_cast
from trulens.utils.typing import OM
from trulens.utils.typing import om_of_many
from trulens.utils.typing import Outputs
from trulens.utils.typing import TensorAKs
from trulens.utils.typing import TensorArgs
from trulens.utils.typing import TensorLike
from trulens.utils.typing import Tensors


class ModelWrapper(AbstractBaseClass):
    """
    A wrapper interface for models that exposes the components needed for 
    computing attributions. This is intended to produce a consistent 
    functionality for all models regardless of the backend/library the model is 
    implemented with.
    """

    @abstractmethod
    def __init__(
        self,
        model,
        *,
        logit_layer=None,
        replace_softmax=False,
        softmax_layer=-1,
        custom_objects=None,
        device=None,
        input_tensors=None,
        output_tensors=None,
        internal_tensor_dict=None,
        default_feed_dict=None,
        session=None,
        **kwargs
    ):
        """
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
        """

        if model is not None:
            self._model = model
        # tf1 backend stores "graph" instead

    @property
    def model(self):
        """
        The model this object wraps.
        """
        return self._model

    def __call__(self, x):
        """
        Shorthand for obtaining the model's output on the given input.

        Parameters:
            x:
                Input point.

        Returns:
            Model's output on the input point.
        """
        return self.fprop(x)[0]

    def fprop(
        self,
        model_args: ArgsLike,
        model_kwargs: KwargsLike = {},
        doi_cut: Optional[Cut] = None,
        to_cut: Optional[Cut] = None,
        attribution_cut: Optional[Cut] = None,
        intervention: InterventionLike = None,
        **kwargs
    ) -> Union[
            ArgsLike[TensorLike],  # attribution_cut is None
            Tuple[ArgsLike[TensorLike],
                  ArgsLike[TensorLike]]  # attribution_cut is not None
    ]:
        """
        **_Used internally by `AttributionMethod`._**

        Several cuts are parameters to this method. These designate how
        information is propagated, intervened on, and returned by this method.
        Models are evaluated starting from model_args, model_kwargs:

            model_inputs -> doi_cut -> attribution_cut -> to_cut

        Tensors for attribution_cut and to_cut are returned. However, the
        returned values are batched according to the batching structure inferred
        from the intervention parameter; interventions are constructed by a DoI
        that product one or more instances per input (e.g.: linear interpolation
        for integrated gradients which produces one instance for each unit of
        resolution). If we have dimensions:

            intervention: (B1, ...) 
            
            model_inputs: (B2, ...)
            
            doi_cut: (B2, ...)

        Then model_inputs are tiled B1/B2 times so that evaluating model on
        tiled model_inputs produces activations at doi_cut of dimension (B1,
        ...) (same as intervention). These activations are replaced by
        intervention.

        Parameters:
            model_args, model_kwargs: 
                The args and kwargs given to the call method of a model. This
                should represent the instances to obtain attributions for,
                assumed to be a *batched* input. if `self.model` supports
                evaluation on *data tensors*, the  appropriate tensor type may
                be used (e.g., Pytorch models may accept Pytorch tensors in
                addition to `np.ndarray`s). The shape of the inputs must match
                the input shape of `self.model`. 

            doi_cut:
                Cut defining where the Distribution of Interest is applied. The
                shape of `intervention` must match the input shape of the
                layer(s) specified by the cut. If `doi_cut` is `None`, the input
                to the model will be used (i.e., `InputCut()`).

            to_cut:
                Cut defining the layer(s) up to which forward propagation should
                be done. This will be the layer over which a quantity of
                interest can be defined. If `to_cut` is `None`, the output of
                the model will be used (i.e., `OutputCut()`). 
            
            attribution_cut:
                Cut defining where the attributions are collected. If
                `attribution_cut` is `None`, it will be assumed to be the
                `doi_cut`. `attribution_cut` should not preceed `doi_cut` in the
                model architecture. 
            
            intervention:
                The intervention created from the Distribution of Interest. If
                `intervention` is `None`, then it is equivalent to the point
                DoI.

            *others*:
                Some backends support additional parameters. Consult the appropriate
                *BACKEND*ModelWrapper._fprop for details.

        Returns:
            list of (backend.Tensor or np.ndarray) (or tuple of lists)
                Lists of output activations are returned, keeping the same type
                as the input. If `attribution_cut` is supplied, also return the
                cut activations.
        """

        if doi_cut is None:
            doi_cut = InputCut()
        if to_cut is None:
            to_cut = OutputCut()

        model_inputs, intervention = self._fprop_organize_vals(
            doi_cut=doi_cut,
            model_args=model_args,
            model_kwargs=model_kwargs,
            intervention=intervention
        )

        B = get_backend()

        # Will cast results to this data container type.
        return_type = type(model_inputs.first_batchable(B))

        rets: Tuple[Outputs[TensorLike], Outputs[TensorLike]] = self._fprop(
            model_inputs=model_inputs,
            doi_cut=doi_cut,
            to_cut=to_cut,
            attribution_cut=attribution_cut,
            intervention=intervention,
            **kwargs
        )
        rets = (to_cut.access_layer(rets[0]), doi_cut.access_layer(rets[1]))
        rets = tuple(
            map(
                lambda ret: om_of_many(
                    nested_cast(backend=B, astype=return_type, args=ret)
                ) if ret is not None else None, rets
            )
        )

        if rets[1] is None:
            return rets[0]
        else:
            return rets

    def _fprop_organize_vals(
        self, *, doi_cut: Cut, model_args: ArgsLike, model_kwargs: KwargsLike,
        intervention: InterventionLike
    ) -> Tuple[ModelInputs, Tensors]:
        """Boundary between public typing of fprop and internal typing in
        _fprop. Converts the variants in the public signature to the specific
        types in the private one. Also handles the logic between input vs.
        non-input cuts and interventions. If doi is input cut, model inputs are
        set from the intervention if available."""

        model_inputs = ModelInputs(many_of_om(model_args), model_kwargs)

        if isinstance(doi_cut, InputCut):
            # For input cuts, produce a ModelInputs container for the intervention and model inputs.
            if intervention is not None:
                if isinstance(intervention, Tensors):
                    # Intervention is using our internal Tensors class. Convert
                    # it to the model inputs class.
                    model_inputs = intervention.as_model_inputs()
                else:
                    # Intervention overrides only args. Using many_of_om here as
                    # sometimes interventions are passed in as single tensors
                    # but args needs a list.
                    intervention = TensorAKs(
                        many_of_om(intervention), model_inputs.kwargs
                    )
                    model_inputs = intervention.as_model_inputs()

                    if len(model_inputs.kwargs) > 0:
                        tru_logger.warning(
                            "Intervention for InputCut DoI specified but contains only positional arguments. "
                            "The rest will be taken from model_kwargs. If you need to intervene on keyword "
                            "arguments, provide the intervention as a ModelInputs container."
                        )

            else:
                # If no intervention given, it is equal to model inputs.
                intervention: ModelInputs = model_inputs

        else:  # doi_cut is not InputCut
            # For non-InputCut, interventions do not have kwargs but for simplifying the logics, we store it
            # in a ModelInputs anyway.

            if intervention is None:
                # Any situations where one wants to specify a non-InputCut intervention with input arguments?
                raise ValueError(
                    "intervention needs to be given for DoI cuts that are not InputCut"
                )
            else:
                # Using many_of_om here as sometimes interventions are passed in as single tensors but args needs a list.
                intervention = TensorArgs(many_of_om(intervention))

        return model_inputs, intervention

    @abstractmethod
    def _fprop(
        self,
        *,
        model_inputs: ModelInputs,  # TensorLike contents only
        doi_cut: Cut,
        to_cut: Cut,
        attribution_cut: Cut,
        intervention: TensorArgs,  # TensorLike contents only
        **kwargs
    ) -> Tuple[Outputs[TensorLike], Outputs[TensorLike]]:
        """Implementation of fprop; arguments, return, and their types are clarified. """

        # Should not have to use DATA_CONTAINER_TYPE internally.

        raise NotImplementedError

    def qoi_bprop(
        self,
        qoi,
        model_args: ArgsLike,
        model_kwargs: KwargsLike = {},
        doi_cut: Optional[Cut] = None,
        to_cut: Optional[Cut] = None,
        attribution_cut: Optional[Cut] = None,
        intervention: InterventionLike = None,
        **kwargs
    ) -> OM[Outputs, OM[Inputs, TensorLike]]:
        """
        **_Used internally by `AttributionMethod`._**
        
        Runs the model beginning at `doi_cut` on input `intervention`, and
        returns the gradients calculated from `to_cut` with respect to
        `attribution_cut` of the quantity of interest.

        Parameters:
            qoi: a Quantity of Interest
                This method will accumulate all gradients of the qoi w.r.t
                `attribution_cut`. 

            model_args, model_kwargs: 
                The args and kwargs given to the call method of a model. This
                should represent the instances to obtain attributions for,
                assumed to be a *batched* input. if `self.model` supports
                evaluation on *data tensors*, the  appropriate tensor type may
                be used (e.g., Pytorch models may accept Pytorch tensors in
                addition to `np.ndarray`s). The shape of the inputs must match
                the input shape of `self.model`. 

            doi_cut:
                Cut defining where the Distribution of Interest is applied. The
                shape of `intervention` must match the input shape of the
                layer(s) specified by the cut. If `doi_cut` is `None`, the input
                to the model will be used (i.e., `InputCut()`).

            to_cut:
                Cut defining the layer(s) at which the propagation will end. The
                If `to_cut` is `None`, the output of the model will be used
                (i.e., `OutputCut()`). `to_cut` cannot preceed `doi_cut` in the
                model architecture, i.e. the gradient of `doi_cut` w.r.t.
                `attribution_cut` must be defined.
            
            attribution_cut:
                Cut defining where the attributions are collected. If
                `attribution_cut` is `None`, it will be assumed to be the
                `doi_cut`.
            
            intervention:
                The intervention created from the Distribution of Interest. If
                `intervention` is `None`, then it is equivalent to the point
                DoI.

            *others*:
                Some backends support additional parameters. Consult the appropriate
                *BACKEND*ModelWrapper._qoi_bprop for details.

        Returns:
            (backend.Tensor or np.ndarray) for each attribution_cut input, for each qoi output
                the gradients of `qoi` w.r.t. `attribution_cut`, keeping same
                type as the input.
                If attribution_cut has multiple inputs, return a list for each. 
                If qoi has multiple outputs, returns a list of the above for each.
        """

        if doi_cut is None:
            doi_cut = InputCut()
        if to_cut is None:
            to_cut = OutputCut()
        if attribution_cut is None:
            attribution_cut = InputCut()

        model_inputs, intervention = self._fprop_organize_vals(
            doi_cut=doi_cut,
            model_args=model_args,
            model_kwargs=model_kwargs,
            intervention=intervention
        )

        # Will cast results to this data container type.
        return_type = type(model_inputs.first_batchable(get_backend()))

        attrs: Outputs[Inputs[TensorLike]] = self._qoi_bprop(
            qoi=qoi,
            model_inputs=model_inputs,
            doi_cut=doi_cut,
            to_cut=to_cut,
            attribution_cut=attribution_cut,
            intervention=intervention,
            **kwargs
        )

        attrs: Outputs[OM[Inputs,
                          TensorLike]] = [om_of_many(attr) for attr in attrs]
        attrs: OM[Outputs, OM[Inputs]] = om_of_many(attrs)

        # Call the implementation and transform its results to the same type as model_inputs.
        return nested_cast(
            backend=get_backend(), astype=return_type, args=attrs
        )

    @abstractmethod
    def _qoi_bprop(
        self, *, qoi: QoI, model_inputs: ModelInputs, doi_cut: Cut, to_cut: Cut,
        attribution_cut: Cut, intervention: TensorArgs, **kwargs
    ) -> Outputs[
            Inputs[TensorLike]
    ]:  # One outer element for each QoI output, one inner element for each attribution_cut input.
        """Implementation of qoi_bprop; arguments, return, and their types are clarified. """
        # Should not have to use DATA_CONTAINER_TYPE internally.

        raise NotImplementedError

    @staticmethod
    def _nested_assign(x, y):
        """
        _nested_assign Assigns tensors values in y to tensors in x.

        Parameters
        ----------
        x:  backend.Tensor or a nested list or tuple of backend.Tensor
            The leaf Tensors will be assigned values from y.
        y:  backend.Tensor or a nested list or tuple of backend.Tensor
            Must be of the same structure as x. Contains objects that
            will be assigned to x.
        """
        if isinstance(y, DATA_CONTAINER_TYPE):
            for i in range(len(y)):
                ModelWrapper._nested_assign(x[i], y[i])
        else:
            try:
                x[:] = y[:]
            except RuntimeError:
                # torch > 1.7.1 does not allow view assignment.
                # We want to keep grads. Using solution from https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308/2
                # Assign directly to Tensor with below.
                x.data = x.data.clone()
                x.data[:] = y[:]

    @staticmethod
    def _flatten(x):
        """
        _flatten Given a nested list or tuple x, outputs a new
            flattened list.

        Parameters
        ----------
        x:  non-collective object or a nested list/tuple of objects
            The nested list to be flattened.
        Returns
        ------
        list
            New list containing the leaf objects in x.

        """
        if isinstance(x, DATA_CONTAINER_TYPE):
            out = []
            for i in range(len(x)):
                out.extend(ModelWrapper._flatten(x[i]))
            return out
        else:
            return [x]

    @staticmethod
    def _unflatten(x, z, count=None):
        """
        _unflatten Given a non-nested list x, outputs a new nested list
            or tuple of the same structure as z.

        Parameters
        ----------
        x:  list of non-collective objects.
            Contains the leaf objects of the nested list.
        z:  non-collective object or a nested list/tuple of objects
            Contains the structure that x will be unflattened to.
        Returns
        ------
        nested list/tuple
            New nested list/tuple containing the leaf objects in x
            and the same structure as z.

        """
        if not count:
            count = [0]
        if isinstance(z, DATA_CONTAINER_TYPE):
            out = []
            for i in range(len(z)):
                out.append(ModelWrapper._unflatten(x, z[i], count))
            return tuple(out) if isinstance(z, tuple) else out
        else:
            out = x[count[0]]
            count[0] += 1
            return out
