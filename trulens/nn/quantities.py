"""
A *Quantity of Interest* (QoI) is a function of the output that determines the 
network output behavior that the attributions describe.

The quantity of interest lets us specify what we want to explain. Often, this is
the output of the network corresponding to a particular class, addressing, e.g.,
"Why did the model classify a given image as a car?" However, we could also 
consider various combinations of outputs, allowing us to ask more specific 
questions, such as, "Why did the model classify a given image as a sedan *and 
not a convertible*?" The former may highlight general “car features,” such as 
tires, while the latter (called a comparative explanation) might focus on the 
roof of the car, a “car feature” not shared by convertibles.
"""
#from __future__ import annotations # Avoid expanding type aliases in mkdocs.

from abc import ABC as AbstractBaseClass
from abc import abstractmethod
from inspect import signature
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

from trulens.nn.backend import get_backend

# Define some type aliases.
TensorLike = Union[Any, List[Union[Any]]]


class QoiCutSupportError(ValueError):
    """
    Exception raised if the quantity of interest is called on a cut whose output
    is not supported by the quantity of interest.
    """
    pass


class QoI(AbstractBaseClass):
    """
    Interface for quantities of interest. The *Quantity of Interest* (QoI) is a
    function of the output specified by the slice that determines the network 
    output behavior that the attributions describe.
    """

    @abstractmethod
    def __call__(self, y: TensorLike) -> TensorLike:
        """
        Computes the distribution of interest from an initial point.

        Parameters:
            y:
                Output point from which the quantity is derived. Must be a
                differentiable tensor.

        Returns:
            A differentiable batched scalar tensor representing the QoI.
        """
        raise NotImplementedError

    def _assert_cut_contains_only_one_tensor(self, x):
        if isinstance(x, list):
            raise QoiCutSupportError(
                'Cut provided to quantity of interest was comprised of '
                'multiple tensors, but `{}` is only defined for cuts comprised '
                'of a single tensor (received a list of {} tensors).\n'
                '\n'
                'Either (1) select a slice where the `to_cut` corresponds to a '
                'single tensor, or (2) implement/use a `QoI` object that '
                'supports lists of tensors, i.e., where the parameter, `x`, to '
                '`__call__` is expected/allowed to be a list of {} tensors.'.
                format(self.__class__.__name__, len(x), len(x)))

        elif not get_backend().is_tensor(x):
            raise ValueError(
                '`{}` expected to receive an instance of `Tensor`, but '
                'received an instance of {}'.format(
                    self.__class__.__name__, type(x)))


class MaxClassQoI(QoI):
    """
    Quantity of interest for attributing output towards the maximum-predicted 
    class.
    """

    def __init__(
            self, axis: int = 1, activation: Union[Callable, str, None] = None):
        """
        Parameters:
            axis:
                Output dimension over which max operation is taken.

            activation:
                Activation function to be applied to the output before taking 
                the max. If `activation` is a string, use the corresponding 
                named activation function implemented by the backend. The 
                following strings are currently supported as shorthands for the
                respective standard activation functions:

                - `'sigmoid'` 
                - `'softmax'` 

                If `activation` is `None`, no activation function is applied to
                the input.
        """
        self._axis = axis
        self.activation = activation

    def __call__(self, y: TensorLike) -> TensorLike:
        self._assert_cut_contains_only_one_tensor(y)

        if self.activation is not None:
            if isinstance(self.activation, str):
                self.activation = self.activation.lower()
                if self.activation in ['sigmoid', 'softmax']:
                    y = getattr(get_backend(), self.activation)(y)

                else:
                    raise NotImplementedError(
                        'This activation function is not currently supported '
                        'by the backend')
            else:
                y = self.activation(y)

        return get_backend().max(y, axis=self._axis)


class InternalChannelQoI(QoI):
    """
    Quantity of interest for attributing output towards the output of an 
    internal convolutional layer channel, aggregating using a specified 
    operation.

    Also works for non-convolutional dense layers, where the given neuron's
    activation is returned.
    """

    @staticmethod
    def _batch_sum(x):
        """
        Sums batched 2D channels, leaving the batch dimension unchanged.
        """
        return get_backend().sum(x, axis=(1, 2))

    def __init__(
            self,
            channel: Union[int, List[int]],
            channel_axis: Optional[int] = None,
            agg_fn: Optional[Callable] = None):
        """
        Parameters:
            channel:
                Channel to return. If a list is provided, then the quantity sums 
                over each of the channels in the list.

            channel_axis:
                Channel dimension index, if relevant, e.g., for 2D convolutional
                layers. If `channel_axis` is `None`, then the channel axis of 
                the relevant backend will be used. This argument is not used 
                when the channels are scalars, e.g., for dense layers.

            agg_fn:
                Function with which to aggregate the remaining dimensions 
                (except the batch dimension) in order to get a single scalar 
                value for each channel. If `agg_fn` is `None` then a sum over 
                each neuron in the channel will be taken. This argument is not 
                used when the channels are scalars, e.g., for dense layers.
        """
        if channel_axis is None:
            channel_axis = get_backend().channel_axis
        if agg_fn is None:
            agg_fn = InternalChannelQoI._batch_sum

        self._channel_ax = channel_axis
        self._agg_fn = agg_fn
        self._channels = channel if isinstance(channel, list) else [channel]

    def __call__(self, y: TensorLike) -> TensorLike:
        B = get_backend()
        self._assert_cut_contains_only_one_tensor(y)

        if len(B.int_shape(y)) == 2:
            return sum([y[:, ch] for ch in self._channels])

        elif len(B.int_shape(y)) == 3:
            return sum([self._agg_fn(y[:, :, ch]) for ch in self._channel])

        elif len(B.int_shape(y)) == 4:
            if self._channel_ax == 1:
                return sum([self._agg_fn(y[:, ch]) for ch in self._channels])

            elif self._channel_ax == 3:
                return sum(
                    [self._agg_fn(y[:, :, :, ch]) for ch in self._channels])

            else:
                raise ValueError(
                    'Unsupported channel axis for convolutional layer: {}'.
                    format(self._channel_ax))

        else:
            raise QoiCutSupportError(
                'Unsupported tensor rank for `InternalChannelQoI`: {}'.format(
                    len(B.int_shape(y))))


class ClassQoI(QoI):
    """
    Quantity of interest for attributing output towards a specified class.
    """

    def __init__(self, cl: int):
        """
        Parameters:
            cl:
                The index of the class the QoI is for.
        """
        self.cl = cl

    def __call__(self, y: TensorLike) -> TensorLike:
        self._assert_cut_contains_only_one_tensor(y)

        return y[:, self.cl]


class ComparativeQoI(QoI):
    """
    Quantity of interest for attributing network output towards a given class, 
    relative to another.
    """

    def __init__(self, cl1: int, cl2: int):
        """
        Parameters:
            cl1:
                The index of the class the QoI is for.
            cl2:
                The index of the class to compare against.
        """
        self.cl1 = cl1
        self.cl2 = cl2

    def __call__(self, y: TensorLike) -> TensorLike:

        self._assert_cut_contains_only_one_tensor(y)

        return y[:, self.cl1] - y[:, self.cl2]


class LambdaQoI(QoI):
    """
    Generic quantity of interest allowing the user to specify a function of the
    model's output as the QoI.
    """

    def __init__(self, function: Callable):
        """
        Parameters:
            function:
                A callable that takes a single argument representing the model's 
                tensor output and returns a differentiable batched scalar tensor 
                representing the QoI.
        """
        if len(signature(function).parameters) != 1:
            raise ValueError(
                'QoI function must take exactly 1 argument, but provided '
                'function takes {} arguments'.format(
                    len(signature(function).parameters)))

        self.function = function

    def __call__(self, y: TensorLike) -> TensorLike:
        return self.function(y)


class ThresholdQoI(QoI):
    """
    Quantity of interest for attributing network output toward the difference 
    between two regions seperated by a given threshold. I.e., the quantity of
    interest is the "high" elements minus the "low" elements, where the high
    elements have activations above the threshold and the low elements have 
    activations below the threshold.

    Use case: bianry segmentation.
    """

    def __init__(
            self,
            threshold: float,
            low_minus_high: bool = False,
            activation: Union[Callable, str, None] = None):
        """
        Parameters:
            threshold:
                A threshold to determine the element-wise sign of the input 
                tensor. The elements with activations higher than the threshold 
                will retain their sign, while the elements with activations 
                lower than the threshold will have their sign flipped (or vice 
                versa if `low_minus_high` is set to `True`).
            low_minus_high:
                If `True`, substract the output with activations above the 
                threshold from the output with activations below the threshold. 
                If `False`, substract the output with activations below the 
                threshold from the output with activations above the threshold.
            activation: str or function, optional
                Activation function to be applied to the quantity before taking
                the threshold. If `activation` is a string, use the 
                corresponding activation function implemented by the backend 
                (currently supported: `'sigmoid'` and `'softmax'`). Otherwise, 
                if `activation` is not `None`, it will be treated as a callable.
                If `activation` is `None`, do not apply an activation function 
                to the quantity.
        """
        # TODO(klas):should this support an aggregation function? By default
        #   this is a sum, but it could, for example, subtract the greatest
        #   positive element from the least negative element.
        self.threshold = threshold
        self.low_minus_high = low_minus_high
        self.activation = activation

    def __call__(self, x: TensorLike) -> TensorLike:
        B = get_backend()
        self._assert_cut_contains_only_one_tensor(x)

        if self.activation is not None:
            if isinstance(self.activation, str):
                self.activation = self.activation.lower()
                if self.activation in ['sigmoid', 'softmax']:
                    x = getattr(B, self.activation)(x)
                else:
                    raise NotImplementedError(
                        'This activation function is not currently supported '
                        'by the backend')
            else:
                x = self.activation(x)

        # TODO(klas): is the `clone` necessary here? Not sure why it was
        #   included.
        mask = B.sign(B.clone(x) - self.threshold)
        if self.low_minus_high:
            mask = -mask

        non_batch_dimensions = tuple(range(len(B.int_shape(x)))[1:])

        return B.sum(mask * x, axis=non_batch_dimensions)


class ClassSeqQoI(QoI):
    """
    Quantity of interest for attributing output towards a sequence of classes 
    for each input.
    """

    def __init__(self, seq_labels: List[int]):
        """
        Parameters:
            seq_labels:
                A sequence of classes corresponding to each input.
        """
        self.seq_labels = seq_labels

    def __call__(self, y):

        self._assert_cut_contains_only_one_tensor(y)
        assert get_backend().shape(y)[0] == len(self.seq_labels)

        return y[:, seq_labels]
