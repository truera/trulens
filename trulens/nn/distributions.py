"""
The distribution of interest lets us specify the set of instances over which we 
want our explanations to be faithful. In some cases, we may want to explain the 
model’s behavior on a particular instance, whereas other times we may be 
interested in a more general behavior over a distribution of instances.
"""
#from __future__ import annotations # Avoid expanding type aliases in mkdocs.

import numpy as np

from abc import ABC as AbstractBaseClass
from abc import abstractmethod
from typing import Any
from typing import List
from typing import Optional
from typing import Union

from trulens.nn.slices import Cut
from trulens.nn import backend as B

# Define some type aliases.
ArrayLike = Union[np.ndarray, Any, List[Union[np.ndarray, Any]]]


class DoiCutSupportError(ValueError):
    """
    Exception raised if the distribution of interest is called on a cut whose
    output is not supported by the distribution of interest.
    """
    pass


class DoI(AbstractBaseClass):
    """
    Interface for distributions of interest. The *Distribution of Interest* 
    (DoI) specifies the instances over which an attribution method is 
    aggregated.
    """

    @abstractmethod
    def __call__(self, z: ArrayLike) -> List[ArrayLike]:
        """
        Computes the distribution of interest from an initial point.

        Parameters:
            z:
                Input point from which the distribution is derived.

        Returns:
            List of points which are all assigned equal probability mass in the
            distribution of interest, i.e., the distribution of interest is
            a discrete, uniform distribution over the list of returned points.
            Each point in the list shares the same type and shape as `z`.
        """
        raise NotImplementedError

    def cut(self) -> Cut:
        """
        Returns:
            The Cut in which the DoI will be applied. If `None`, the DoI will be
            applied to the input. otherwise, the distribution should be applied
            to the latent space defined by the cut. The default implementation 
            of this method always returns `None`; unless this method is 
            overridden, the DoI can only be applied to the input.
        """
        return None

    def get_activation_multiplier(self, activation: ArrayLike) -> ArrayLike:
        """
        Returns a term to multiply the gradient by to convert from "*influence 
        space*" to "*attribution space*". Conceptually, "influence space"
        corresponds to the potential effect of a slight increase in each 
        feature, while "attribution space" corresponds to an approximation of
        the net marginal contribution to the quantity of interest of each 
        feature.

        Parameters:
            activation:
                The activation of the layer the DoI is applied to.

        Returns:
            An array with the same shape as ``activation`` that will be 
            multiplied by the gradient to obtain the attribution. The default 
            implementation of this method simply returns ``activation``.
        """
        return activation

    def _assert_cut_contains_only_one_tensor(self, x):
        if isinstance(x, list):
            raise DoiCutSupportError(
                '\n\n'
                'Cut provided to distribution of interest was comprised of '
                'multiple tensors, but `{}` is only defined for cuts comprised '
                'of a single tensor (received a list of {} tensors).\n'
                '\n'
                'Either (1) select a slice where the `to_cut` corresponds to a '
                'single tensor, or (2) implement/use a `DoI` object that '
                'supports lists of tensors, i.e., where the parameter, `z`, to '
                '`__call__` is expected/allowed to be a list of {} tensors.'.
                format(self.__class__.__name__, len(x), len(x)))

        elif not (isinstance(x, np.ndarray) or isinstance(x, B.Tensor)):
            raise ValueError(
                '`{}` expected to receive an instance of `B.Tensor` or '
                '`np.ndarray`, but received an instance of {}'.format(
                    self.__class__.__name__, type(x)))


class PointDoi(DoI):
    """
    Distribution that puts all probability mass on a single point.
    """

    def __call__(self, z):
        return [z]


class LinearDoi(DoI):
    """
    Distribution representing the linear interpolation between a baseline and 
    the given point. Used by Integrated Gradients.
    """

    def __init__(
            self,
            baseline: Optional[ArrayLike] = None,
            resolution: int = 10,
            cut: Cut = None):
        """
        The DoI for point, `z`, will be a uniform distribution over the points
        on the line segment connecting `z` to `baseline`, approximated by a
        sample of `resolution` points equally spaced along this segment.

        Parameters:
            baseline:
                The baseline to interpolate from. Must be same shape as the 
                space the distribution acts over, i.e., the shape of the points, 
                `z`, eventually passed to `__call__`. If `cut` is `None`, this
                must be the same shape as the input, otherwise this must be the
                same shape as the latent space defined by the cut. If `None` is
                given, `baseline` will be the zero vector in the appropriate 
                shape.

            resolution:
                Number of points returned by each call to this DoI. A higher 
                resolution is more computationally expensive, but gives a better
                approximation of the DoI this object mathematically represents.

            cut:
                If None, the DoI will be applied to the input. Otherwise, the 
                DoI will be applied to the Cut. This allows a DoI to be applied
                in the latent space.
        """
        self._baseline = baseline
        self._resolution = resolution
        self._cut = cut

    def __call__(self, z: ArrayLike) -> List[ArrayLike]:
        if isinstance(z, (list, tuple)) and len(z) == 1:
            z = z[0]

        self._assert_cut_contains_only_one_tensor(z)

        if self._baseline is None:
            baseline = B.zeros_like(z)
        else:
            baseline = self._baseline

        if (B.is_tensor(z) and not B.is_tensor(baseline)):
            baseline = B.as_tensor(baseline)

        if (not B.is_tensor(z) and B.is_tensor(baseline)):
            baseline = B.as_array(baseline)

        r = 1. if self._resolution is 1 else self._resolution - 1.

        return [
            (1. - i / r) * z + i / r * baseline
            for i in range(self._resolution)
        ]

    def cut(self) -> Cut:
        """
        Returns:
            A cut to apply the distribution supplied in the constructor.
            if None, the DoI will apply to the input.
        """
        return self._cut

    def get_activation_multiplier(self, activation: ArrayLike) -> ArrayLike:
        """
        Returns a term to multiply the gradient by to convert from "*influence 
        space*" to "*attribution space*". Conceptually, "influence space"
        corresponds to the potential effect of a slight increase in each 
        feature, while "attribution space" corresponds to an approximation of
        the net marginal contribution to the quantity of interest of each 
        feature.

        Parameters:
            activation:
                The activation of the layer the DoI is applied to.

        Returns:
            The activation adjusted by the baseline passed to the constructor.
        """
        return (
            activation if self._baseline is None else activation -
            self._baseline)


class GaussianDoi(DoI):
    """
    Distribution representing a Gaussian ball around the point. Used by Smooth
    Gradients.
    """

    def __init__(self, var: float, resolution: int):
        """
        Parameters:
            var:
                The variance of the Gaussian noise to be added around the point.

            resolution:
                Number of samples returned by each call to this DoI.
        """
        self._var = var
        self._resolution = resolution

    def __call__(self, z: ArrayLike) -> List[ArrayLike]:

        self._assert_cut_contains_only_one_tensor(z)

        if B.is_tensor(z):
            # Tensor implementation.
            return [
                z + B.random_normal_like(z, var=self._var)
                for _ in range(self._resolution)
            ]

        else:
            # Array implementation.
            return [
                z + np.random.normal(0., np.sqrt(self._var), z.shape)
                for _ in range(self._resolution)
            ]
