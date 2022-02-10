"""
The distribution of interest lets us specify the set of samples over which we 
want our explanations to be faithful. In some cases, we may want to explain the 
model’s behavior on a particular record, whereas other times we may be 
interested in a more general behavior over a distribution of samples.
"""
#from __future__ import annotations # Avoid expanding type aliases in mkdocs.

from abc import ABC as AbstractBaseClass
from abc import abstractmethod
from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from trulens.nn.backend import get_backend
from trulens.nn.slices import Cut
from trulens.utils.typing import DATA_CONTAINER_TYPE, ArrayLike, BaselineLike, ModelInputs


class DoiCutSupportError(ValueError):
    """
    Exception raised if the distribution of interest is called on a cut whose
    output is not supported by the distribution of interest.
    """
    pass


class DoI(AbstractBaseClass):
    """
    Interface for distributions of interest. The *Distribution of Interest* 
    (DoI) specifies the samples over which an attribution method is 
    aggregated.
    """

    def __init__(self, cut: Cut = None):
        """"Initialize DoI

        Parameters:
            cut (Cut, optional): 
                The Cut in which the DoI will be applied. If `None`, the DoI will be
                applied to the input. otherwise, the distribution should be applied
                to the latent space defined by the cut. 
        """
        self._cut = cut

    @abstractmethod
    def __call__(
            self,
            z: ArrayLike,
            *,
            model_inputs: Optional[ModelInputs] = None) -> List[ArrayLike]:
        """
        Computes the distribution of interest from an initial point.

        Parameters:
            z:
                Input point from which the distribution is derived.
            model_inputs:
                Optional wrapped model input arguments that produce value z at
                cut.

        Returns:
            List of points which are all assigned equal probability mass in the
            distribution of interest, i.e., the distribution of interest is a
            discrete, uniform distribution over the list of returned points.
            Each point in the list shares the same type and shape as `z`.
        """
        raise NotImplementedError

    # @property
    def cut(self) -> Cut:
        """
        Returns:
            The Cut in which the DoI will be applied. If `None`, the DoI will be
            applied to the input. otherwise, the distribution should be applied
            to the latent space defined by the cut. 
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

        elif not (isinstance(x, np.ndarray) or get_backend().is_tensor(x)):
            raise ValueError(
                '`{}` expected to receive an instance of `Tensor` or '
                '`np.ndarray`, but received an instance of {}'.format(
                    self.__class__.__name__, type(x)))


class PointDoi(DoI):
    """
    Distribution that puts all probability mass on a single point.
    """

    def __init__(self, cut: Cut = None):
        """"Initialize PointDoI

        Parameters:
            cut (Cut, optional): 
                The Cut in which the DoI will be applied. If `None`, the DoI will be
                applied to the input. otherwise, the distribution should be applied
                to the latent space defined by the cut. 
        """
        super(PointDoi, self).__init__(cut)

    def __call__(self,
                 z,
                 *,
                 model_inputs: Optional[ModelInputs] = None) -> List[ArrayLike]:

        return [z]


class LinearDoi(DoI):
    """
    Distribution representing the linear interpolation between a baseline and 
    the given point. Used by Integrated Gradients.
    """

    def __init__(
        self,
        baseline: BaselineLike = None,
        resolution: int = 10,
        *,
        cut: Cut = None,
    ):
        """
        The DoI for point, `z`, will be a uniform distribution over the points
        on the line segment connecting `z` to `baseline`, approximated by a
        sample of `resolution` points equally spaced along this segment.

        Parameters:
            cut (Cut, optional, from DoI): 
                The Cut in which the DoI will be applied. If `None`, the DoI
                will be applied to the input. otherwise, the distribution should
                be applied to the latent space defined by the cut. 
            baseline (optional)
                The baseline to interpolate from. Must be same shape as the
                space the distribution acts over, i.e., the shape of the points,
                `z`, eventually passed to `__call__`. If `cut` is `None`, this
                must be the same shape as the input, otherwise this must be the
                same shape as the latent space defined by the cut. If `None` is
                given, `baseline` will be the zero vector in the appropriate
                shape. If the baseline is callable, it is expected to return the
                `baseline`, given `z` and optional model arguments.
            resolution (int):
                Number of points returned by each call to this DoI. A higher
                resolution is more computationally expensive, but gives a better
                approximation of the DoI this object mathematically represents.
        """
        super(LinearDoi, self).__init__(cut)
        self._baseline = baseline
        self._resolution = resolution

    @property
    def baseline(self) -> ArrayLike:
        return self._baseline

    @property
    def resolution(self) -> int:
        return self._resolution

    def __call__(
            self,
            z: ArrayLike,
            *,
            model_inputs: Optional[ModelInputs] = None) -> List[ArrayLike]:

        if isinstance(z, DATA_CONTAINER_TYPE) and len(z) == 1:
            z = z[0]

        self._assert_cut_contains_only_one_tensor(z)

        baseline = self._compute_baseline(z, model_inputs=model_inputs)

        r = 1. if self._resolution is 1 else self._resolution - 1.

        return [
            (1. - i / r) * z + i / r * baseline
            for i in range(self._resolution)
        ]

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

    def _compute_baseline(
            self,
            z: ArrayLike,
            *,
            model_inputs: Optional[ModelInputs] = None) -> ArrayLike:

        B = get_backend()

        _baseline = self.baseline

        if isinstance(_baseline, Callable):
            num_arguments = len(signature(_baseline).parameters)
            if num_arguments == 1:
                _baseline = _baseline(z)
            elif num_arguments == 2:
                _baseline = _baseline(z, model_inputs)
            else:
                raise RuntimeError(
                    f"Baseline function has an unexected signature: {signature(_baseline)}."
                )

        if _baseline is None:
            _baseline = B.zeros_like(z)

        if B.is_tensor(z) and not B.is_tensor(_baseline):
            _baseline = B.as_tensor(_baseline)

        if not B.is_tensor(z) and B.is_tensor(_baseline):
            _baseline = B.as_array(_baseline)

        return _baseline


class GaussianDoi(DoI):
    """
    Distribution representing a Gaussian ball around the point. Used by Smooth
    Gradients.
    """

    def __init__(self, var: float, resolution: int, cut: Cut = None):
        """
        Parameters:
            var:
                The variance of the Gaussian noise to be added around the point.

            resolution:
                Number of samples returned by each call to this DoI.
            cut (Cut, optional): 
                The Cut in which the DoI will be applied. If `None`, the DoI will be
                applied to the input. otherwise, the distribution should be applied
                to the latent space defined by the cut. 
        """
        super(GaussianDoi, self).__init__(cut)
        self._var = var
        self._resolution = resolution

    def __call__(self, z: ArrayLike) -> List[ArrayLike]:
        B = get_backend()
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
