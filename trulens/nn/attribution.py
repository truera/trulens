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

import numpy as np

from abc import ABC as AbstractBaseClass
from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from trulens.nn.distributions import DoI
from trulens.nn.distributions import LinearDoi
from trulens.nn.distributions import PointDoi
from trulens.nn.models import ModelWrapper
from trulens.nn.models._model_base import DATA_CONTAINER_TYPE
from trulens.nn.quantities import ComparativeQoI
from trulens.nn.quantities import InternalChannelQoI
from trulens.nn.quantities import QoI
from trulens.nn.quantities import LambdaQoI
from trulens.nn.quantities import MaxClassQoI
from trulens.nn.slices import Cut
from trulens.nn.slices import InputCut
from trulens.nn.slices import OutputCut
from trulens.nn.slices import Slice
from trulens.nn import backend as B

# Define some type aliases.
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
    def __init__(self, model: ModelWrapper, *args, **kwargs):
        """
        Abstract constructor.

        Parameters:
            model :
                Model for which attributions are calculated.
        """
        self._model = model

    @property
    def model(self) -> ModelWrapper:
        """
        Model for which attributions are calculated.
        """
        return self._model

    @abstractmethod
    def attributions(self, *model_args, **model_kwargs):
        """
        Returns attributions for the given input.

        Parameters:
            model_args, model_kwargs: 
                The args and kwargs given to the call method of a model.
                This should represent the instances to obtain attributions for, 
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
    specifies the instances over which the attributions are aggregated.
    
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
            multiply_activation: bool = True):
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
        self.doi = InternalInfluence.__get_doi(doi)
        self._do_multiply = multiply_activation

    def attributions(self, *model_args, **model_kwargs):
        doi_cut = self.doi.cut() if self.doi.cut() else InputCut()

        doi_val = self.model.fprop(model_args, model_kwargs, to_cut=doi_cut)

        # DoI supports tensor or list of tensor. unwrap args to perform DoI on
        # top level list

        # Depending on the model_arg input, the data may be nested in data
        # containers. We unwrap so that there operations are working on a single
        # level of data container.
        if isinstance(doi_val, DATA_CONTAINER_TYPE) and isinstance(
                doi_val[0], DATA_CONTAINER_TYPE):
            doi_val = doi_val[0]

        if isinstance(doi_val, DATA_CONTAINER_TYPE) and len(doi_val) == 1:
            doi_val = doi_val[0]

        D = self.doi(doi_val)
        n_doi = len(D)
        D = InternalInfluence.__concatenate_doi(D)

        # Calculate the gradient of each of the points in the DoI.
        qoi_grads = self.model.qoi_bprop(
            self.qoi,
            model_args,
            model_kwargs,
            attribution_cut=self.slice.from_cut,
            to_cut=self.slice.to_cut,
            intervention=D,
            doi_cut=doi_cut)
        # Take the mean across the samples in the DoI.
        if isinstance(qoi_grads, DATA_CONTAINER_TYPE):
            attributions = [
                B.mean(
                    B.reshape(qoi_grad, (n_doi, -1) + qoi_grad.shape[1:]),
                    axis=0) for qoi_grad in qoi_grads
            ]
        else:
            attributions = B.mean(
                B.reshape(qoi_grads, (n_doi, -1) + qoi_grads.shape[1:]), axis=0)

        # Multiply by the activation multiplier if specified.
        if self._do_multiply:
            z_val = self.model.fprop(
                model_args, model_kwargs, to_cut=self.slice.from_cut)
            if isinstance(z_val, DATA_CONTAINER_TYPE) and len(z_val) == 1:
                z_val = z_val[0]

            if isinstance(attributions, DATA_CONTAINER_TYPE):
                for i in range(len(attributions)):
                    if isinstance(z_val, DATA_CONTAINER_TYPE) and len(
                            z_val) == len(attributions):
                        attributions[i] *= self.doi.get_activation_multiplier(
                            z_val[i])
                    else:
                        attributions[i] *= (
                            self.doi.get_activation_multiplier(z_val))

            else:
                attributions *= self.doi.get_activation_multiplier(z_val)

        return attributions

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
            if len(qoi_arg) is 2:
                return ComparativeQoI(*qoi_arg)

            else:
                raise ValueError(
                    'Tuple or list argument for `qoi` must have length 2')

        elif isinstance(qoi_arg, str):
            # We can specify `MaxClassQoI` via the string 'max'.
            if qoi_arg == 'max':
                return MaxClassQoI()

            else:
                raise ValueError(
                    'String argument for `qoi` must be one of the following:\n'
                    '  - "max"')

        else:
            raise ValueError('Unrecognized argument type for `qoi`')

    @staticmethod
    def __get_doi(doi_arg):
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
                return PointDoi()

            elif doi_arg == 'linear':
                return LinearDoi()

            else:
                raise ValueError(
                    'String argument for `doi` must be one of the following:\n'
                    '  - "point"\n'
                    '  - "linear"')

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
            if len(slice_arg) is 2:
                return Slice(
                    InternalInfluence.__get_cut(slice_arg[0]),
                    InternalInfluence.__get_cut(slice_arg[1]))

            else:
                raise ValueError(
                    'Tuple or list argument for `cuts` must have length 2')

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
                'least one point.')

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
                D = [B.as_array(d) for d in D]
            return np.concatenate(D)


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
            cut: CutLike = None,
            qoi: QoiLike = 'max',
            doi: DoiLike = 'point',
            multiply_activation: bool = True):
        """
        Parameters:
            model :
                Model for which attributions are calculated.

            cut :
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

            doi : distributions.DoI | str
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

            multiply_activation : bool, optional
                Whether to multiply the gradient result by its corresponding
                activation, thus converting from "*influence space*" to 
                "*attribution space*."
        """
        super().__init__(
            model, (InputCut(), cut),
            qoi,
            doi,
            multiply_activation=multiply_activation)


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
            resolution: int = 50):
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
        super().__init__(
            model,
            OutputCut(),
            'max',
            LinearDoi(baseline, resolution),
            multiply_activation=True)
