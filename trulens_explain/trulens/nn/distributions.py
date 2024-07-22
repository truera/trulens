"""
The distribution of interest lets us specify the set of samples over which we
want our explanations to be faithful. In some cases, we may want to explain the
model’s behavior on a particular record, whereas other times we may be
interested in a more general behavior over a distribution of samples.
"""
#from __future__ import annotations # Avoid expanding type aliases in mkdocs.

from abc import ABC as AbstractBaseClass
from abc import abstractmethod
from typing import Callable, Optional

import numpy as np
from trulens.nn.backend import get_backend
from trulens.nn.slices import Cut
from trulens.utils.typing import accepts_model_inputs
from trulens.utils.typing import BaselineLike
from trulens.utils.typing import DATA_CONTAINER_TYPE
from trulens.utils.typing import Inputs
from trulens.utils.typing import many_of_om
from trulens.utils.typing import MAP_CONTAINER_TYPE
from trulens.utils.typing import ModelInputs
from trulens.utils.typing import nested_cast
from trulens.utils.typing import nested_map
from trulens.utils.typing import nested_zip
from trulens.utils.typing import OM
from trulens.utils.typing import om_of_many
from trulens.utils.typing import render_object
from trulens.utils.typing import TensorAKs
from trulens.utils.typing import TensorLike
from trulens.utils.typing import Uniform


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

    def __str__(self):
        return render_object(self, ['_cut'])

    def _wrap_public_call(
        self, z: Inputs[TensorLike], *, model_inputs: ModelInputs
    ) -> Inputs[Uniform[TensorLike]]:
        """Same as __call__ but input and output types are more specific and
        less permissive. Formats the inputs for special cases that might be more
        convenient for the user's __call__ implementation and formats its return
        back to the consistent type."""

        z: Inputs[TensorLike] = om_of_many(z)

        if accepts_model_inputs(self.__call__):
            ret = self.__call__(z, model_inputs=model_inputs)
        else:
            ret = self.__call__(z)
        # Wrap the public doi generator with appropriate type aliases.
        if isinstance(ret, DATA_CONTAINER_TYPE):
            if isinstance(ret[0], DATA_CONTAINER_TYPE):
                ret = Inputs(Uniform(x) for x in ret)
            else:
                ret = Uniform(ret)

            ret: Inputs[Uniform[TensorLike]] = many_of_om(
                ret, innertype=Uniform
            )
        else:
            ret: ArgsLike = [ret]
        return ret

    @abstractmethod
    def __call__(
        self,
        z: OM[Inputs, TensorLike],
        *,
        model_inputs: Optional[ModelInputs] = None
    ) -> OM[Inputs, Uniform[TensorLike]]:
        """
        Computes the distribution of interest from an initial point. If z:
        TensorLike is given, we assume there is only 1 input to the DoI layer. If
        z: List[TensorLike] is given, it provides all of the inputs to the DoI
        layer.

        Either way, we always return List[List[TensorLike]] (alias
        Inputs[Uniform[TensorLike]]) with outer list spanning layer inputs, and
        inner list spanning a distribution's instance.

        Parameters:
            z:
                Input point from which the distribution is derived. If
                list/tuple, the point is defined by multiple tensors.
            model_inputs:
                Optional wrapped model input arguments that produce value z at
                cut.

        Returns:
            List of points which are all assigned equal probability mass in the
            distribution of interest, i.e., the distribution of interest is a
            discrete, uniform distribution over the list of returned points. If
            z is multi-input, returns a distribution for each input.
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

    def _wrap_public_get_activation_multiplier(
        self, activation: Inputs[TensorLike], *, model_inputs: ModelInputs
    ) -> Inputs[TensorLike]:
        """Same as get_activation_multiplier but without "one-or-more". """

        activations: OM[Inputs, TensorLike] = om_of_many(activation)

        # get_activation_multiplier is public
        if accepts_model_inputs(self.get_activation_multiplier):
            ret: OM[Inputs, TensorLike] = self.get_activation_multiplier(
                activations, model_inputs=model_inputs
            )
        else:
            ret: OM[Inputs,
                    TensorLike] = self.get_activation_multiplier(activations)

        ret: Inputs[TensorLike] = many_of_om(ret)

        return ret

    def get_activation_multiplier(
        self,
        activation: OM[Inputs, TensorLike],
        *,
        model_inputs: Optional[ModelInputs] = None
    ) -> OM[Inputs, TensorLike]:
        """
        Returns a term to multiply the gradient by to convert from "*influence
        space*" to "*attribution space*". Conceptually, "influence space"
        corresponds to the potential effect of a slight increase in each
        feature, while "attribution space" corresponds to an approximation of
        the net marginal contribution to the quantity of interest of each
        feature.

        Parameters:
            activation:
                The activation of the layer the DoI is applied to. DoI may be
                multi-input in which case activation will be a list.
            model_inputs:
                Optional wrapped model input arguments that produce activation
                at cut.

        Returns:
            An array with the same shape as ``activation`` that will be
            multiplied by the gradient to obtain the attribution. The default
            implementation of this method simply returns ``activation``. If
            activation is multi-input, returns one multiplier for each.
        """
        return om_of_many(activation)

    def _assert_cut_contains_only_one_tensor(self, x):
        if isinstance(x, DATA_CONTAINER_TYPE) and len(x) == 1:
            x = x[0]
        if isinstance(x, MAP_CONTAINER_TYPE) and len(x) == 1:
            x = list(x.values())[0]

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
                format(self.__class__.__name__, len(x), len(x))
            )

        elif not (isinstance(x, np.ndarray) or get_backend().is_tensor(x)):
            raise ValueError(
                '`{}` expected to receive an instance of `Tensor` or '
                '`np.ndarray`, but received an instance of {}'.format(
                    self.__class__.__name__, type(x)
                )
            )


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

    def __call__(
        self,
        z: OM[Inputs, TensorLike],
        *,
        model_inputs: Optional[ModelInputs] = None
    ) -> OM[Inputs, Uniform[TensorLike]]:

        z: Inputs[TensorLike] = many_of_om(z)

        return om_of_many(nested_map(z, lambda x: [x]))


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
            baseline (BaselineLike, optional):
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
    def baseline(self) -> BaselineLike:
        return self._baseline

    @property
    def resolution(self) -> int:
        return self._resolution

    def __str__(self):
        return render_object(self, ['_cut', '_baseline', '_resolution'])

    def __call__(
        self,
        z: OM[Inputs, TensorLike],
        *,
        model_inputs: Optional[ModelInputs] = None
    ) -> OM[Inputs, Uniform[TensorLike]]:

        self._assert_cut_contains_only_one_tensor(z)

        z: Inputs[TensorLike] = many_of_om(z)

        baseline = self._compute_baseline(z, model_inputs=model_inputs)

        r = 1. if self._resolution == 1 else self._resolution - 1.
        zipped = nested_zip(z, baseline)

        def zipped_interpolate(zipped_z_baseline):
            """interpolates zipped elements

            Args:
                zipped_z_baseline: A tuple expecting the first element to be the z_val, and second to be the baseline.

            Returns:
                a list of interpolations from z to baseline
            """
            z_ = zipped_z_baseline[0]
            b_ = zipped_z_baseline[1]
            return [ # Uniform
                (1. - i / r) * z_ + i / r * b_
                for i in range(self._resolution)
            ]

        ret = om_of_many(
            nested_map(
                zipped, zipped_interpolate, check_accessor=lambda x: x[0]
            )
        )

        return ret

    def get_activation_multiplier(
        self,
        activation: OM[Inputs, TensorLike],
        *,
        model_inputs: Optional[ModelInputs] = None
    ) -> Inputs[TensorLike]:
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

        activation: Inputs[TensorLike] = many_of_om(activation)

        baseline: Inputs[TensorLike] = self._compute_baseline(
            activation, model_inputs=model_inputs
        )

        if baseline is None:
            return activation

        zipped = nested_zip(activation, baseline)

        def zipped_subtract(zipped_activation_baseline):
            """subtracts zipped elements

            Args:
                zipped_activation_baseline: A tuple expecting the first element to be the activation, and second to be the baseline.

            Returns:
                a subtraction of activation and baseline
            """
            activation = zipped_activation_baseline[0]
            baseline = zipped_activation_baseline[1]
            return activation - baseline

        ret = nested_map(zipped, zipped_subtract, check_accessor=lambda x: x[0])
        return ret

    def _compute_baseline(
        self,
        z: Inputs[TensorLike],
        *,
        model_inputs: Optional[ModelInputs] = None
    ) -> Inputs[TensorLike]:

        B = get_backend()

        _baseline: BaselineLike = self.baseline  # user-provided

        if isinstance(_baseline, Callable):
            if accepts_model_inputs(_baseline):
                _baseline: OM[Inputs, TensorLike] = many_of_om(
                    _baseline(om_of_many(z), model_inputs=model_inputs)
                )
            else:
                _baseline: OM[Inputs, TensorLike] = many_of_om(
                    _baseline(om_of_many(z))
                )

        else:
            _baseline: OM[Inputs, TensorLike]

        if _baseline is None:
            _baseline: Inputs[TensorLike] = nested_map(z, B.zeros_like)
        else:
            _baseline: Inputs[TensorLike] = many_of_om(_baseline)
            # Came from user; could have been single or multiple inputs.
        tensor_wrapper = TensorAKs(args=z)
        # Cast to either Tensor or numpy.ndarray to match what was given in z.
        return nested_cast(
            backend=B,
            args=_baseline,
            astype=type(tensor_wrapper.first_batchable(B))
        )


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

    def __str__(self):
        return render_object(self, ['_cut', '_var', '_resolution'])

    def __call__(self, z: OM[Inputs,
                             TensorLike]) -> OM[Inputs, Uniform[TensorLike]]:
        # Public interface.

        B = get_backend()
        self._assert_cut_contains_only_one_tensor(z)

        def gauss_of_input(z: TensorLike) -> Uniform[TensorLike]:
            # TODO: make a pytorch backend with the same interface to use in places like these.

            if B.is_tensor(z):
                # Tensor implementation.
                return [
                    z + B.random_normal_like(z, var=self._var)
                    for _ in range(self._resolution)
                ]  # Uniform

            else:
                # Array implementation.
                return [
                    z + np.random.normal(0., np.sqrt(self._var), z.shape)
                    for _ in range(self._resolution)
                ]  # Uniform

        z: Inputs[TensorLike] = many_of_om(z)

        return om_of_many(nested_map(z, gauss_of_input))
