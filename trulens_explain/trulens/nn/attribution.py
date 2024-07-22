"""
*Attribution methods* quantitatively measure the contribution of each of a
function's individual inputs to its output. Gradient-based attribution methods
compute the gradient of a model with respect to its inputs to describe how
important each input is towards the output prediction. These methods can be
applied to assist in explaining deep networks.

TruLens provides implementations of several such techniques, found in this
package.
"""
#from __future__ import annotations # Avoid expanding type aliases in mkdocs.

from abc import ABC as AbstractBaseClass
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, get_type_hints, List, Tuple, Union

import numpy as np
from trulens.nn.backend import get_backend
from trulens.nn.backend import memory_suggestions
from trulens.nn.backend import rebatch
from trulens.nn.backend import tile
from trulens.nn.distributions import DoI
from trulens.nn.distributions import LinearDoi
from trulens.nn.distributions import PointDoi
from trulens.nn.models._model_base import ModelWrapper
from trulens.nn.quantities import ComparativeQoI
from trulens.nn.quantities import InternalChannelQoI
from trulens.nn.quantities import LambdaQoI
from trulens.nn.quantities import MaxClassQoI
from trulens.nn.quantities import QoI
from trulens.nn.slices import Cut
from trulens.nn.slices import InputCut
from trulens.nn.slices import OutputCut
from trulens.nn.slices import Slice
from trulens.utils import tru_logger
from trulens.utils.typing import ArgsLike
from trulens.utils.typing import DATA_CONTAINER_TYPE
from trulens.utils.typing import Inputs
from trulens.utils.typing import KwargsLike
from trulens.utils.typing import many_of_om
from trulens.utils.typing import MAP_CONTAINER_TYPE
from trulens.utils.typing import ModelInputs
from trulens.utils.typing import nested_axes
from trulens.utils.typing import nested_cast
from trulens.utils.typing import nested_map
from trulens.utils.typing import nested_zip
from trulens.utils.typing import OM
from trulens.utils.typing import om_of_many
from trulens.utils.typing import Outputs
from trulens.utils.typing import TensorArgs
from trulens.utils.typing import TensorLike
from trulens.utils.typing import Uniform

# Attribution-related type aliases.
# TODO: Verify these and move to typing utils?
CutLike = Union[Cut, int, str, None]
SliceLike = Union[Slice, Tuple[CutLike], CutLike]
QoiLike = Union[QoI, int, Tuple[int], Callable, str]
DoiLike = Union[DoI, str]


@dataclass
class AttributionResult:
    """
    _attribution method output container.
    """

    attributions: Outputs[Inputs[TensorLike]] = None
    gradients: Outputs[Inputs[Uniform[TensorLike]]] = None
    interventions: Inputs[Uniform[TensorLike]] = None


# Order of dimensions for multi-dimensional or nested containers. See more
# information in typing.py.
AttributionResult.axes = {
    attribute: nested_axes(typ)
    for attribute, typ in get_type_hints(AttributionResult).items()
}


class AttributionMethod(AbstractBaseClass):
    """
    Interface used by all attribution methods.

    An attribution method takes a neural network model and provides the ability
    to assign values to the variables of the network that specify the importance
    of each variable towards particular predictions.
    """

    @abstractmethod
    def __init__(
        self, model: ModelWrapper, rebatch_size: int = None, *args, **kwargs
    ):
        """
        Abstract constructor.

        Parameters:
            model: ModelWrapper
                Model for which attributions are calculated.

            rebatch_size: int (optional)
                Will rebatch instances to this size if given. This may be
                required for GPU usage if using a DoI which produces multiple
                instances per user-provided instance. Many valued DoIs will
                expand the tensors sent to each layer to original_batch_size *
                doi_size. The rebatch size will break up original_batch_size *
                doi_size into rebatch_size chunks to send to model.
        """
        self._model = model

        self.rebatch_size = rebatch_size

    @property
    def model(self) -> ModelWrapper:
        """
        Model for which attributions are calculated.
        """
        return self._model

    @abstractmethod
    def _attributions(self, model_inputs: ModelInputs) -> AttributionResult:
        """
        For attributions that have options to return multiple things depending
        on configuration, wrap those multiple things in the AttributionResult
        tuple.
        """
        ...

    def attributions(
        self, *model_args: ArgsLike, **model_kwargs: KwargsLike
    ) -> Union[TensorLike, ArgsLike[TensorLike],
               ArgsLike[ArgsLike[TensorLike]]]:
        """
        Returns attributions for the given input. Attributions are in the same
        shape as the layer that attributions are being generated for.

        The numeric scale of the attributions will depend on the specific
        implementations of the Distribution of Interest and Quantity of
        Interest. However it is generally related to the scale of gradients on
        the Quantity of Interest.

        For example, Integrated Gradients uses the linear interpolation
        Distribution of Interest which subsumes the completeness axiom which
        ensures the sum of all attributions of a record equals the output
        determined by the Quantity of Interest on the same record.

        The Point Distribution of Interest will be determined by the gradient at
        a single point, thus being a good measure of model sensitivity.

        Parameters:
            model_args: ArgsLike, model_kwargs: KwargsLike
                The args and kwargs given to the call method of a model. This
                should represent the records to obtain attributions for, assumed
                to be a *batched* input. if `self.model` supports evaluation on
                *data tensors*, the  appropriate tensor type may be used (e.g.,
                Pytorch models may accept Pytorch tensors in addition to
                `np.ndarray`s). The shape of the inputs must match the input
                shape of `self.model`.

        Returns
            - np.ndarray when single attribution_cut input, single qoi output
            - or ArgsLike[np.ndarray] when single input, multiple output (or
              vice versa)
            - or ArgsLike[ArgsLike[np.ndarray]] when multiple output (outer),
              multiple input (inner)

            An array of attributions, matching the shape and type of `from_cut`
            of the slice. Each entry in the returned array represents the degree
            to which the corresponding feature affected the model's outcome on
            the corresponding point.

            If attributing to a component with multiple inputs, a list for each
            will be returned.

            If the quantity of interest features multiple outputs, a list for
            each will be returned.
        """

        # Calls like: attributions([arg1, arg2]) will get read as model_args =
        # ([arg1, arg2],), that is, a tuple with a single element containing the
        # model args. Test below checks for this. TODO: Disallow such
        # invocations? They should be given as attributions(arg1, arg2).
        if isinstance(model_args,
                      tuple) and len(model_args) == 1 and isinstance(
                          model_args[0], DATA_CONTAINER_TYPE):
            model_args = model_args[0]

        model_inputs = ModelInputs(
            args=many_of_om(model_args), kwargs=model_kwargs
        )
        # Will cast results to this data container type.
        return_type = type(model_inputs.first_batchable(get_backend()))

        pieces = self._attributions(model_inputs)

        # Format attributions into the public structure which throws out output
        # lists and input lists if there is only one output or only one input.
        # Also cast to whatever the input type was.
        attributions: Outputs[Inputs[np.ndarray]] = nested_cast(
            backend=get_backend(), astype=return_type, args=pieces.attributions
        )
        attributions: Outputs[OM[Inputs, np.ndarray]
                             ] = [om_of_many(attr) for attr in attributions]
        attributions: OM[Outputs, OM[Inputs,
                                     np.ndarray]] = om_of_many(attributions)

        if pieces.gradients is not None or pieces.interventions is not None:
            tru_logger.warning(
                'AttributionMethod configured to return gradients or interventions. '
                'Use the internal _attribution call to retrieve those.'
            )

        return attributions


class InternalInfluence(AttributionMethod):
    """Internal attributions parameterized by a slice, quantity of interest, and
    distribution of interest.

    The *slice* specifies the layers at which the internals of the model are to
    be exposed; it is represented by two *cuts*, which specify the layer the
    attributions are assigned to and the layer from which the quantity of
    interest is derived. The *Quantity of Interest* (QoI) is a function of the
    output specified by the slice that determines the network output behavior
    that the attributions are to describe. The *Distribution of Interest* (DoI)
    specifies the records over which the attributions are aggregated.

    More information can be found in the following paper:

    [Influence-Directed Explanations for Deep Convolutional Networks](
        https://arxiv.org/pdf/1802.03788.pdf)

    This should be cited using:

    ```bibtex
    @INPROCEEDINGS{
        leino18influence,
        author={
            Klas Leino and
            Shayak Sen and
            Anupam Datta and
            Matt Fredrikson and
            Linyi Li},
        title={
            Influence-Directed Explanations
            for Deep Convolutional Networks},
        booktitle={IEEE International Test Conference (ITC)},
        year={2018},
    }
    ```
    """

    def __init__(
        self,
        model: ModelWrapper,
        cuts: SliceLike,
        qoi: QoiLike,
        doi: DoiLike,
        multiply_activation: bool = True,
        return_grads: bool = False,
        return_doi: bool = False,
        *args,
        **kwargs
    ):
        """
        Parameters:
            model:
                Model for which attributions are calculated.

            cuts:
                The slice to use when computing the attributions. The slice
                keeps track of the layer whose output attributions are
                calculated and the layer for which the quantity of interest is
                computed. Expects a `Slice` object, or a related type that can
                be interpreted as a `Slice`, as documented below.

                If a single `Cut` object is given, it is assumed to be the cut
                representing the layer for which attributions are calculated
                (i.e., `from_cut` in `Slice`) and the layer for the quantity of
                interest (i.e., `to_cut` in `slices.Slice`) is taken to be the
                output of the network. If a tuple or list of two `Cut`s is
                given, they are assumed to be `from_cut` and `to_cut`,
                respectively.

                A cut (or the cuts within the tuple) can also be represented as
                an `int`, `str`, or `None`. If an `int` is given, it represents
                the index of a layer in `model`. If a `str` is given, it
                represents the name of a layer in `model`. `None` is an
                alternative for `slices.InputCut`.

            qoi:
                Quantity of interest to attribute. Expects a `QoI` object, or a
                related type that can be interpreted as a `QoI`, as documented
                below.

                If an `int` is given, the quantity of interest is taken to be
                the slice output for the class/neuron/channel specified by the
                given integer, i.e.,
                ```python
                quantities.InternalChannelQoI(qoi)
                ```

                If a tuple or list of two integers is given, then the quantity
                of interest is taken to be the comparative quantity for the
                class given by the first integer against the class given by the
                second integer, i.e.,
                ```python
                quantities.ComparativeQoI(*qoi)
                ```

                If a callable is given, it is interpreted as a function
                representing the QoI, i.e.,
                ```python
                quantities.LambdaQoI(qoi)
                ```

                If the string, `'max'`, is given, the quantity of interest is
                taken to be the output for the class with the maximum score,
                i.e.,
                ```python
                quantities.MaxClassQoI()
                ```

            doi:
                Distribution of interest over inputs. Expects a `DoI` object, or
                a related type that can be interpreted as a `DoI`, as documented
                below.

                If the string, `'point'`, is given, the distribution is taken to
                be the single point passed to `attributions`, i.e.,
                ```python
                distributions.PointDoi()
                ```

                If the string, `'linear'`, is given, the distribution is taken
                to be the linear interpolation from the zero input to the point
                passed to `attributions`, i.e.,
                ```python
                distributions.LinearDoi()
                ```

            multiply_activation:
                Whether to multiply the gradient result by its corresponding
                activation, thus converting from "*influence space*" to
                "*attribution space*."
        """
        super().__init__(model, *args, **kwargs)

        self.slice = InternalInfluence.__get_slice(cuts)
        self.qoi = InternalInfluence.__get_qoi(qoi)
        self.doi = InternalInfluence.__get_doi(doi, cut=self.slice.from_cut)
        self._do_multiply = multiply_activation
        self._return_grads = return_grads
        self._return_doi = return_doi

    def _attributions(self, model_inputs: ModelInputs) -> AttributionResult:
        # NOTE: not symbolic

        B = get_backend()
        results = AttributionResult()

        # Create a message for out-of-memory errors regarding float and batch size.
        first_batchable = model_inputs.first_batchable(B)
        if first_batchable is None:
            batch_size = 1
        else:
            batch_size = first_batchable.shape[0]

        param_msgs = [
            f'float size = {B.floatX_size} ({B.floatX}); consider changing to a smaller type.',
            f'batch size = {batch_size}; consider reducing the size of the batch you send to the attributions method.'
        ]

        doi_cut = self.doi.cut() if self.doi.cut() else InputCut()

        with memory_suggestions(*param_msgs):  # Handles out-of-memory messages.
            doi_val: List[B.Tensor] = self.model._fprop(
                model_inputs=model_inputs,
                to_cut=doi_cut,
                doi_cut=InputCut(),
                attribution_cut=None,  # InputCut(),
                intervention=model_inputs
            )[0]

        doi_val = nested_map(doi_val, B.as_array)

        D = self.doi._wrap_public_call(doi_val, model_inputs=model_inputs)

        if self._return_doi:
            results.interventions = D  # : Inputs[Uniform[TensorLike]]

        D_tensors = D[0]
        n_doi = len(D_tensors)
        if isinstance(D_tensors, MAP_CONTAINER_TYPE):
            for k in D_tensors.keys():
                if isinstance(D_tensors[k], DATA_CONTAINER_TYPE):
                    n_doi = len(D_tensors[k])
        D = self.__concatenate_doi(D)
        rebatch_size = self.rebatch_size
        if rebatch_size is None:
            rebatch_size = len(D[0])

        intervention = TensorArgs(args=D)
        model_inputs_expanded = tile(what=model_inputs, onto=intervention)
        # Create a message for out-of-memory errors regarding doi_size.
        # TODO: Generalize this message to doi other than LinearDoI:
        doi_size_msg = f'distribution of interest size = {n_doi}; consider reducing intervention resolution.'

        combined_batch_size = n_doi * batch_size
        combined_batch_msg = f'combined batch size = {combined_batch_size}; consider reducing batch size, intervention size'

        rebatch_size_msg = f'rebatch_size = {rebatch_size}; consider reducing this AttributionMethod constructor parameter (default is same as combined batch size).'

        # Calculate the gradient of each of the points in the DoI.
        with memory_suggestions(
                param_msgs +
            [doi_size_msg, combined_batch_msg, rebatch_size_msg]
        ):  # Handles out-of-memory messages.
            qoi_grads_expanded: List[Outputs[Inputs[TensorLike]]] = []

            for inputs_batch, intervention_batch in rebatch(
                    model_inputs_expanded, intervention,
                    batch_size=rebatch_size):

                qoi_grads_expanded_batch: Outputs[
                    Inputs[TensorLike]] = self.model._qoi_bprop(
                        qoi=self.qoi,
                        model_inputs=inputs_batch,
                        attribution_cut=self.slice.from_cut,
                        to_cut=self.slice.to_cut,
                        intervention=intervention_batch,
                        doi_cut=doi_cut
                    )

                # important to cast to numpy inside loop:
                qoi_grads_expanded.append(
                    nested_map(qoi_grads_expanded_batch, B.as_array)
                )

        num_outputs = len(qoi_grads_expanded[0])
        num_inputs = len(qoi_grads_expanded[0][0])
        transpose = [
            [[] for _ in range(num_inputs)] for _ in range(num_outputs)
        ]
        for o in range(num_outputs):
            for i in range(num_inputs):
                for qoi_grads_batch in qoi_grads_expanded:
                    transpose[o][i].append(qoi_grads_batch[o][i])

        def container_concat(x):
            """Applies np concatenate on a container. If it is a map type, it will apply it on each key.

            Args:
                x (map or data container): A container of tensors

            Returns:
                concatenated tensors of the container.
            """
            if isinstance(x[0], MAP_CONTAINER_TYPE):
                ret_map = {}
                for k in x[0].keys():
                    ret_map[k] = np.concatenate([_dict[k] for _dict in x])
                return ret_map
            else:
                return np.concatenate(x)

        qoi_grads_expanded: Outputs[Inputs[np.ndarray]] = nested_map(
            transpose, container_concat, nest=2
        )
        qoi_grads_expanded: Outputs[Inputs[np.ndarray]] = nested_map(
            qoi_grads_expanded,
            lambda grad: np.reshape(grad, (n_doi, -1) + grad.shape[1:]),
            nest=2
        )
        if self._return_grads:
            results.gradients = qoi_grads_expanded  # : Outputs[Inputs[Uniform[TensorLike]]]

        # TODO: Does this need to be done in numpy?
        attrs: Outputs[Inputs[TensorLike]] = nested_map(
            qoi_grads_expanded, lambda grad: np.mean(grad, axis=0), nest=2
        )

        # Multiply by the activation multiplier if specified.
        if self._do_multiply:
            with memory_suggestions(param_msgs):
                z_val = self.model._fprop(
                    model_inputs=model_inputs,
                    doi_cut=InputCut(),
                    attribution_cut=None,
                    to_cut=self.slice.from_cut,
                    intervention=model_inputs  # intentional
                )[0]

            mults: Inputs[TensorLike
                         ] = self.doi._wrap_public_get_activation_multiplier(
                             z_val, model_inputs=model_inputs
                         )
            mults: Inputs[np.ndarray] = nested_cast(
                backend=B, args=mults, astype=np.ndarray
            )
            mult_attrs = []
            for attr in attrs:  # Outputs

                zipped = nested_zip(attr, mults)

                def zip_mult(zipped_attr_mults):
                    attr = zipped_attr_mults[0]
                    mults = zipped_attr_mults[1]
                    return attr * mults

                attr = nested_map(
                    zipped, zip_mult, check_accessor=lambda x: x[0]
                )
                mult_attrs.append(attr)
            attrs = mult_attrs
        results.attributions = attrs  # : Outputs[Inputs[TensorLike]]

        return results

    @staticmethod
    def __get_qoi(qoi_arg):
        """
        Helper function to get a `QoI` object from more user-friendly primitive
        arguments.
        """
        # TODO(klas): we could potentially do some basic error catching here,
        #   for example, making sure the index for a given channel is in range.

        if isinstance(qoi_arg, QoI):
            # We were already given a QoI, so return it.
            return qoi_arg

        elif callable(qoi_arg):
            # If we were given a callable, treat that function as a QoI.
            return LambdaQoI(qoi_arg)

        elif isinstance(qoi_arg, int):
            # If we receive an int, we take it to be the class/channel index
            # (whether it's a class or channel depends on the layer the quantity
            # is for, but `InternalChannelQoI` generalizes to both).
            return InternalChannelQoI(qoi_arg)

        elif isinstance(qoi_arg, DATA_CONTAINER_TYPE):
            # If we receive a DATA_CONTAINER_TYPE, we take it to be two classes
            # for which we are performing a comparative quantity of interest.
            if len(qoi_arg) == 2:
                return ComparativeQoI(*qoi_arg)

            else:
                raise ValueError(
                    'Tuple or list argument for `qoi` must have length 2'
                )

        elif isinstance(qoi_arg, str):
            # We can specify `MaxClassQoI` via the string 'max'.
            if qoi_arg == 'max':
                return MaxClassQoI()

            else:
                raise ValueError(
                    'String argument for `qoi` must be one of the following:\n'
                    '  - "max"'
                )

        else:
            raise ValueError('Unrecognized argument type for `qoi`')

    @staticmethod
    def __get_doi(doi_arg, cut=None):
        """
        Helper function to get a `DoI` object from more user-friendly primitive
        arguments.
        """
        if isinstance(doi_arg, DoI):
            # We were already given a DoI, so return it.
            return doi_arg

        elif isinstance(doi_arg, str):
            # We can specify `PointDoi` via the string 'point', or `LinearDoi`
            # via the string 'linear'.
            if doi_arg == 'point':
                return PointDoi(cut=cut)

            elif doi_arg == 'linear':
                return LinearDoi(cut=cut)

            else:
                raise ValueError(
                    'String argument for `doi` must be one of the following:\n'
                    '  - "point"\n'
                    '  - "linear"'
                )

        else:
            raise ValueError('Unrecognized argument type for `doi`')

    @staticmethod
    def __get_slice(slice_arg):
        """
        Helper function to get a `Slice` object from more user-friendly
        primitive arguments.
        """
        if isinstance(slice_arg, Slice):
            # We are already given a Slice, so return it.
            return slice_arg

        elif (isinstance(slice_arg, Cut) or isinstance(slice_arg, int) or
              isinstance(slice_arg, str) or slice_arg is None or
              slice_arg == 0):

            # If we receive a Cut, we take it to be the Cut of the start layer.
            return Slice(InternalInfluence.__get_cut(slice_arg), OutputCut())

        elif isinstance(slice_arg, DATA_CONTAINER_TYPE):
            # If we receive a DATA_CONTAINER_TYPE, we take it to be the start
            # and end layer of the slice.
            if len(slice_arg) == 2:
                if slice_arg[1] is None:
                    return Slice(
                        InternalInfluence.__get_cut(slice_arg[0]), OutputCut()
                    )
                else:
                    return Slice(
                        InternalInfluence.__get_cut(slice_arg[0]),
                        InternalInfluence.__get_cut(slice_arg[1])
                    )

            else:
                raise ValueError(
                    'Tuple or list argument for `cuts` must have length 2'
                )

        else:
            raise ValueError('Unrecognized argument type for `cuts`')

    @staticmethod
    def __get_cut(cut_arg):
        """
        Helper function to get a `Cut` object from more user-friendly primitive
        arguments.
        """
        if isinstance(cut_arg, Cut):
            # We are already given a Cut, so return it.
            return cut_arg

        elif cut_arg is None or cut_arg == 0:
            # If we receive None or zero, we take it to be the input cut.
            return InputCut()

        # TODO(klas): may want a bit more validation here.
        elif isinstance(cut_arg, int) or isinstance(cut_arg, str):
            return Cut(cut_arg)

        else:
            raise ValueError('Unrecognized argument type for cut')

    @staticmethod
    def __concatenate_doi(D: Inputs[Uniform[TensorLike]]) -> Inputs[TensorLike]:
        # Returns one TensorLike for each model input.
        if len(D[0]) == 0:
            raise ValueError(
                'Got empty distribution of interest. `DoI` must return at '
                'least one point.'
            )
        # TODO: should this always be done in numpy or can we do it in backend?
        D = nested_cast(backend=get_backend(), args=D, astype=np.ndarray)
        ret = nested_map(D, np.concatenate, nest=1)
        return ret


class InputAttribution(InternalInfluence):
    """
    Attributions of input features on either internal or output quantities. This
    is essentially an alias for

    ```python
    InternalInfluence(
        model,
        (trulens.nn.slices.InputCut(), cut),
        qoi,
        doi,
        multiply_activation)
    ```
    """

    def __init__(
        self,
        model: ModelWrapper,
        qoi_cut: CutLike = None,  # see WARNING-LOAD-INIT
        qoi: QoiLike = 'max',
        doi_cut: CutLike = None,  # see WARNING-LOAD-INIT
        doi: DoiLike = 'point',
        multiply_activation: bool = True,
        *args,
        **kwargs
    ):
        """
        Parameters:
            model :
                Model for which attributions are calculated.

            qoi_cut :
                The cut determining the layer from which the QoI is derived.
                Expects a `Cut` object, or a related type that can be
                interpreted as a `Cut`, as documented below.

                If an `int` is given, it represents the index of a layer in
                `model`.

                If a `str` is given, it represents the name of a layer in
                `model`.

                `None` is an alternative for `slices.OutputCut()`.

            qoi : quantities.QoI | int | tuple | str
                Quantity of interest to attribute. Expects a `QoI` object, or a
                related type that can be interpreted as a `QoI`, as documented
                below.

                If an `int` is given, the quantity of interest is taken to be
                the slice output for the class/neuron/channel specified by the
                given integer, i.e., ```python
                quantities.InternalChannelQoI(qoi) ```

                If a tuple or list of two integers is given, then the quantity
                of interest is taken to be the comparative quantity for the
                class given by the first integer against the class given by the
                second integer, i.e., ```python quantities.ComparativeQoI(*qoi)
                ```

                If a callable is given, it is interpreted as a function
                representing the QoI, i.e., ```python quantities.LambdaQoI(qoi)
                ```

                If the string, `'max'`, is given, the quantity of interest is
                taken to be the output for the class with the maximum score,
                i.e., ```python quantities.MaxClassQoI() ```

            doi_cut :
                For models which have non-differentiable pre-processing at the
                start of the model, specify the cut of the initial
                differentiable input form. For NLP models, for example, this
                could point to the embedding layer. If not provided, InputCut is
                assumed.

            doi : distributions.DoI | str
                Distribution of interest over inputs. Expects a `DoI` object, or
                a related type that can be interpreted as a `DoI`, as documented
                below.

                If the string, `'point'`, is given, the distribution is taken to
                be the single point passed to `attributions`, i.e., ```python
                distributions.PointDoi() ```

                If the string, `'linear'`, is given, the distribution is taken
                to be the linear interpolation from the zero input to the point
                passed to `attributions`, i.e., ```python
                distributions.LinearDoi() ```

            multiply_activation : bool, optional
                Whether to multiply the gradient result by its corresponding
                activation, thus converting from "*influence space*" to
                "*attribution space*."
        """
        if doi_cut is None:
            # WARNING-LOAD-INIT: Do not put this as a default arg in the def
            # line. That would cause an instantiation of InputCut when this
            # class is loaded and before it is used. Because get_backend gets
            # called in Cut.__init__, it may fail if this class is loaded before
            # trulens.nn.models.get_model_wrapper is called on some model.
            doi_cut = InputCut()

        super().__init__(
            model, (doi_cut, qoi_cut),
            qoi,
            doi,
            multiply_activation=multiply_activation,
            *args,
            **kwargs
        )


class IntegratedGradients(InputAttribution):
    """
    Implementation for the Integrated Gradients method from the following paper:

    [Axiomatic Attribution for Deep Networks](
        https://arxiv.org/pdf/1703.01365)

    This should be cited using:

    ```bibtex
    @INPROCEEDINGS{
        sundararajan17axiomatic,
        author={Mukund Sundararajan and Ankur Taly, and Qiqi Yan},
        title={Axiomatic Attribution for Deep Networks},
        booktitle={International Conference on Machine Learning (ICML)},
        year={2017},
    }
    ```

    This is essentially an alias for

    ```python
    InternalInfluence(
        model,
        (trulens.nn.slices.InputCut(), trulens.nn.slices.OutputCut()),
        'max',
        trulens.nn.distributions.LinearDoi(baseline, resolution),
        multiply_activation=True)
    ```
    """

    def __init__(
        self,
        model: ModelWrapper,
        baseline=None,
        resolution: int = 50,
        doi_cut=None,  # see WARNING-LOAD-INIT
        qoi='max',
        qoi_cut=None,  # see WARNING-LOAD-INIT
        *args,
        **kwargs
    ):
        """
        Parameters:
            model:
                Model for which attributions are calculated.

            baseline:
                The baseline to interpolate from. Must be same shape as the
                input. If `None` is given, the zero vector in the appropriate
                shape will be used.

            resolution:
                Number of points to use in the approximation. A higher
                resolution is more computationally expensive, but gives a better
                approximation of the mathematical formula this attribution
                method represents.
        """

        if doi_cut is None:
            doi_cut = InputCut()

        if qoi_cut is None:
            qoi_cut = OutputCut()

        super().__init__(
            model=model,
            qoi_cut=qoi_cut,
            qoi=qoi,
            doi_cut=doi_cut,
            doi=LinearDoi(baseline, resolution, cut=doi_cut),
            multiply_activation=True,
            *args,
            **kwargs
        )
