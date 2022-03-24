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
from collections import defaultdict
from typing import Callable, Iterable, List, Tuple, Union
import sys

import numpy as np

from trulens.nn.backend import get_backend
from trulens.nn.backend import grace
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
from trulens.utils.typing import accepts_model_inputs, argslike_cast, as_container
from trulens.utils.typing import as_args
from trulens.utils.typing import DATA_CONTAINER_TYPE
from trulens.utils.typing import ModelInputs

# Attribution-related type aliases.
CutLike = Union[Cut, int, str, None]
SliceLike = Union[Slice, Tuple[CutLike], CutLike]
QoiLike = Union[QoI, int, Tuple[int], Callable, str]
DoiLike = Union[DoI, str]


class AttributionMethod(AbstractBaseClass):
    """
    Interface used by all attribution methods.
    
    An attribution method takes a neural network model and provides the ability
    to assign values to the variables of the network that specify the importance
    of each variable towards particular predictions.
    """

    @abstractmethod
    def __init__(self, model: ModelWrapper, doi_per_batch=None, *args, **kwargs):
        """
        Abstract constructor.

        Parameters:
            model :
                Model for which attributions are calculated.
        """
        self._model = model

        self.doi_per_batch = doi_per_batch

    @property
    def model(self) -> ModelWrapper:
        """
        Model for which attributions are calculated.
        """
        return self._model

    @abstractmethod
    def attributions(self, *model_args, **model_kwargs):
        """
        Returns attributions for the given input. Attributions are in the same shape
        as the layer that attributions are being generated for. 
        
        The numeric scale of the attributions will depend on the specific implementations 
        of the Distribution of Interest and Quantity of Interest. However it is generally 
        related to the scale of gradients on the Quantity of Interest. 

        For example, Integrated Gradients uses the linear interpolation Distribution of Interest
        which subsumes the completeness axiom which ensures the sum of all attributions of a record
        equals the output determined by the Quantity of Interest on the same record. 

        The Point Distribution of Interest will be determined by the gradient at a single point,
        thus being a good measure of model sensitivity. 

        Parameters:
            model_args, model_kwargs: 
                The args and kwargs given to the call method of a model.
                This should represent the records to obtain attributions for, 
                assumed to be a *batched* input. if `self.model` supports
                evaluation on *data tensors*, the  appropriate tensor type may
                be used (e.g., Pytorch models may accept Pytorch tensors in 
                addition to `np.ndarray`s). The shape of the inputs must match
                the input shape of `self.model`. 

        Returns:
            An array of attributions, matching the shape and type of `from_cut`
            of the slice. Each entry in the returned array represents the degree
            to which the corresponding feature affected the model's outcome on
            the corresponding point.
        """
        raise NotImplementedError


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
        multiply_activation: bool = True
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
        super().__init__(model)

        self.slice = InternalInfluence.__get_slice(cuts)
        self.qoi = InternalInfluence.__get_qoi(qoi)
        self.doi = InternalInfluence.__get_doi(doi, cut=self.slice.from_cut)
        self._do_multiply = multiply_activation

    def attributions(self, *model_args, **model_kwargs):
        # NOTE: not symbolic

        B = get_backend()

        model_inputs = ModelInputs(model_args, model_kwargs)

        # Will cast results to this data container type.
        return_type = type(model_inputs.first())

        # From now on, data containers, whether B.Tensor or np.ndarray will be consistent.
        # model_inputs = model_inputs.map(lambda t: t if B.is_tensor(t) else B.as_tensor(t))

        print("attributions:")
        print("model_inputs=", model_inputs)

        # Create a message for out-of-memory errors regarding float and batch size.
        if len(list(model_inputs.values())) == 0:
            batch_size = 1
        else:
            batch_size = model_inputs.first().shape[0]

        param_msgs = [
            f"float size = {B.floatX_size} ({B.floatX}); consider changing to a smaller type.",
            f"batch size = {batch_size}; consider reducing the size of the batch you send to the attributions method."
        ]

        doi_cut = self.doi.cut() if self.doi.cut() else InputCut()

        print("doi_cut=", doi_cut)

        with grace(*param_msgs): # Handles out-of-memory messages.
            doi_val: List[B.Tensor] = self.model._fprop(
                model_inputs=model_inputs,
                to_cut=doi_cut,
                doi_cut=InputCut(),
                attribution_cut=None,# InputCut(),
                intervention=model_inputs
            )

        # DoI supports tensor or list of tensor. unwrap args to perform DoI on
        # top level list

        # Depending on the model_arg input, the data may be nested in data
        # containers. We unwrap so that there operations are working on a single
        # level of data container.
        # TODO(piotrm): automatic handling of other common containers like the ones
        # used for huggingface models.
        # TODO(piotrm): this unwrapping may not be necessary any more
        # doi_val = as_container(doi_val)

        # doi_val = list(map(B.as_array, doi_val))

        #if isinstance(doi_val, DATA_CONTAINER_TYPE) and isinstance(
        #        doi_val[0], DATA_CONTAINER_TYPE):
        #    doi_val = doi_val[0]

        #if isinstance(doi_val, DATA_CONTAINER_TYPE) and len(doi_val) == 1:
        #    doi_val = doi_val[0]

        # assert(len(doi_val) == 1)

        print("doi_val=", doi_val)
        # doi_val = list(map(B.as_array, doi_val))

        #if isinstance(doi_val, DATA_CONTAINER_TYPE) and len(doi_val) == 1:
        #    doi_val = doi_val[0]

        print("doi_val=", doi_val)

        if accepts_model_inputs(self.doi):
            D = self.doi(doi_val, model_inputs=model_inputs)
        else:
            D = self.doi(doi_val)

        # D = D[0]
        n_doi = len(D)
        doi_per_batch = self.doi_per_batch
        if self.doi_per_batch is None:
            doi_per_batch = n_doi

        print("D pre concat=", D)
        #D_ = []
        #for batch in self.__concatenate_doi(D):
        #    if isinstance(batch, DATA_CONTAINER_TYPE):
        #        D_.append(list(map(B.as_tensor, batch)))
        #    else:
        #        D_.append(B.as_tensor(batch))

        
        #D = D

        D = self.__concatenate_doi(D)

        # D = list(map(B.as_tensor, D))#, doi_per_batch=doi_per_batch)

        # D = B.as_tensor(D)

        #if isinstance(D, DATA_CONTAINER_TYPE):
        #    D = list(map(B.as_tensor, D))
        #else:
        #    D = B.as_tensor(D)

        print("D post concat=", D)
  
        # Create a message for out-of-memory errors regarding doi_size.
        # TODO: Generalize this message to doi other than LinearDoI:
        doi_size_msg = f"distribution of interest size = {n_doi}; consider reducing intervention resolution."
        doi_per_batch_msg = f"doi_per_batch = {doi_per_batch}; consider reducing this (default is same as the above)."

        # TODO: Consider doing the model_inputs tiling here instead of inside qoi_bprop.

        effective_batch_size = doi_per_batch * batch_size
        effective_batch_msg = f"effective batch size = {effective_batch_size}; consider reducing batch size, intervention size, or doi_per_batch"

        # Calculate the gradient of each of the points in the DoI.
        # qoi_grads_per_arg = defaultdict(list)
        with grace(param_msgs + [doi_size_msg, doi_per_batch_msg, effective_batch_msg]): # Handles out-of-memory messages.
            #for Dbatch in D:
            #    print("Dbatch=", Dbatch)

                #qoi_grads_for_all_args = 
            qoi_grads = self.model._qoi_bprop(
                qoi=self.qoi,
                #model_args=model_inputs.args,
                #model_kwargs=model_inputs.kwargs,
                model_inputs=model_inputs,
                attribution_cut=self.slice.from_cut,
                to_cut=self.slice.to_cut,
                #intervention=D,
                intervention = ModelInputs(D, {}),
                doi_cut=doi_cut
            )
                #print("qoi_grads_for_all_args=", len(qoi_grads_for_all_args), type(qoi_grads_for_all_args))
                #for arg_index, qoi_grads_for_arg in enumerate(qoi_grads_for_all_args):
                #    qoi_grads_per_arg[arg_index].append(qoi_grads_for_arg)

                    # print("qoi_grads_=", qoi_grads)

        #qoi_grads = [np.concatenate(qoi_grads_for_arg) for qoi_grads_for_arg in qoi_grads_per_arg.values()]
        #print("len(qoi_grads)=", len(qoi_grads), type(qoi_grads))
        #if len(qoi_grads) == 1:
        #    qoi_grads = qoi_grads[0]
        #if len(qoi_grads) == 1:
        #    qoi_grads = qoi_grads[0]
        #print("qoi_grads=", qoi_grads)

        # Take the mean across the samples in the DoI.

        # assert(len(qoi_grads) > 0)

        print("qoi_grads=", qoi_grads)

        if isinstance(qoi_grads, DATA_CONTAINER_TYPE):
            qoi_grads = list(map(B.as_array, qoi_grads))
            # TODO: Below is done in numpy.
            attributions = [
                B.mean(
                    B.reshape(qoi_grad, (n_doi, -1) + qoi_grad.shape[1:]),
                    axis=0
                ) for qoi_grad in qoi_grads
            ]
        else:
            # raise ValueError("inconsistent")
            # TODO: Below is actually done in numpy, not backend.
            qoi_grads = B.as_array(qoi_grads)
            attributions = B.mean(
                B.reshape(qoi_grads, (n_doi, -1) + qoi_grads.shape[1:]), axis=0
            )

        # print("attributions=", attributions.shape)

        extra_args = dict()
        if accepts_model_inputs(self.doi.get_activation_multiplier):
            extra_args['model_inputs'] = model_inputs

        # Multiply by the activation multiplier if specified.
        if self._do_multiply:
            with grace(param_msgs):
                z_val = self.model._fprop(
                    model_inputs=model_inputs,
                    doi_cut=InputCut(),
                    attribution_cut=None,
                    to_cut=self.slice.from_cut,
                    intervention=model_inputs
                )

            if isinstance(z_val, DATA_CONTAINER_TYPE) and len(z_val) == 1:
                z_val = z_val[0]

            # print("z_val=", z_val.shape)

            if isinstance(attributions, DATA_CONTAINER_TYPE):
                for i in range(len(attributions)):
                    if isinstance(z_val, DATA_CONTAINER_TYPE) and len(
                            z_val) == len(attributions):
                        attributions[i] *= self.doi.get_activation_multiplier(
                            z_val[i], **extra_args
                        )
                    else:
                        attributions[i] *= (
                            self.doi.get_activation_multiplier(
                                z_val, **extra_args
                            )
                        )

            else:
                # raise ValueError("inconsistent")
                attributions *= self.doi.get_activation_multiplier(
                    z_val, **extra_args
                )

        # Cast to the same data type as provided inputs.
        return argslike_cast(
            backend=B,
            astype=return_type,
            args=attributions
        )

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
    def __concatenate_doi(D):
        if len(D) == 0:
            raise ValueError(
                'Got empty distribution of interest. `DoI` must return at '
                'least one point.'
            )

        if isinstance(D[0], DATA_CONTAINER_TYPE):
            transposed = [[] for _ in range(len(D[0]))]
            for point in D:
                for i, v in enumerate(point):
                    transposed[i].append(v)

            return [
                np.concatenate(D_i)
                if isinstance(D_i[0], np.ndarray) else D_i[0]
                for D_i in transposed
            ]

        else:
            if not isinstance(D[0], np.ndarray):
                D = [get_backend().as_array(d) for d in D]
            return np.concatenate(D)


    def __concatenate_doi2(self, D, doi_per_batch):
        if len(D) == 0:
            raise ValueError(
                'Got empty distribution of interest. `DoI` must return at '
                'least one point.'
            )

        # Why are there these two cases?

        doi_per_batch = len(D)

        # DoI provides multiple values (i.e. for interventions for model inputs with multiple arguments).
        if isinstance(D[0], DATA_CONTAINER_TYPE):
            print("case1")
            print("len(D)=", len(D))
            print("len(D[0])=", len(D[0]))

            batches = []
            for i in range(0, len(D), doi_per_batch):

                Di = D[i:i+doi_per_batch]

                # number of arguments for model inputs
                num_args = len(Di[0])

                vals_per_arg = [[] for _ in range(num_args)]
                for point in Di:
                    for arg_index, v in enumerate(point):
                        vals_per_arg[arg_index].append(v)# if isinstance(v, np.ndarray) else v[0])

                #batches.append(
                #    np.concatenate(D_i)
                #    if isinstance(D_i[0], np.ndarray) else D_i[0]
                #    for D_i in transposed
                #)

                batches.append([np.concatenate(vals_for_arg) for vals_for_arg in vals_per_arg])

            return batches[0]

        else:
            print("case2")
            print("len(D)=", len(D))
            print("D[0]=", D[0].shape)           

            if not isinstance(D[0], np.ndarray):
                D = [get_backend().as_array(d) for d in D]

            batches = []
            for i in range(0, len(D), doi_per_batch):
                batches.append(
                    [np.concatenate(D[i:i+doi_per_batch])] # outer 1-item list indicates a single argument intervention
                )

            #return np.concatenate(D)
            return batches[0]


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
        multiply_activation: bool = True
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
            multiply_activation=multiply_activation
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
        qoi_cut=None  # see WARNING-LOAD-INIT
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
            multiply_activation=True
        )
