""" 
One clear use case for measuring attributions is for human consumption. In order
to be fully leveraged by humans, explanations need to be interpretable &mdash; a
large vector of numbers doesn’t in general make us more confident we understand
what a network is doing. We therefore view an *explanation* as comprised of both
an *attribution measurement* and an *interpretation* of what the attribution 
values represent.

One obvious way to interpret attributions, particularly in the image domain, is
via visualization. This module provides several visualization methods for
interpreting attributions as images.
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
import tempfile
from tkinter import N
from typing import Any, Callable, Iterable, List, Optional, Set, Tuple, TypeVar, Union

import numpy as np

from trulens.nn.attribution import AttributionMethod
from trulens.nn.attribution import InternalInfluence
from trulens.nn.backend import get_backend
from trulens.nn.distributions import PointDoi
from trulens.nn.models._model_base import ModelWrapper
from trulens.nn.quantities import ClassQoI, ComparativeQoI, InternalChannelQoI, MaxClassQoI, QoI
from trulens.nn.slices import Cut
from trulens.nn.slices import InputCut
from trulens.utils import tru_logger
from trulens.utils import try_import
from trulens.utils.typing import Inputs, ModelInputs, Outputs, TensorLike, Uniform, nested_cast
from trulens.utils.typing import Tensor
from trulens.utils.typing import Tensors

# TODO: Unify the common things across image and NLP visualizations.


class Tiler(object):
    """
    Used to tile batched images or attributions.
    """

    def tile(self, a: np.ndarray) -> np.ndarray:
        """
        Tiles the given array into a grid that is as square as possible.

        Parameters:
            a:
                An array of 4D batched image data.

        Returns:
            A tiled array of the images from `a`. The resulting array has rank
            3 for color images, and 2 for grayscale images (the batch dimension
            is removed, as well as the channel dimension for grayscale images).
            The resulting array has its color channel dimension ordered last to
            fit the requirements of the `matplotlib` library.
        """

        # `pyplot` expects the channels to come last.
        if get_backend().dim_order == 'channels_first':
            a = a.transpose((0, 2, 3, 1))

        n, h, w, c = a.shape

        rows = int(np.sqrt(n))
        cols = int(np.ceil(float(n) / rows))

        new_a = np.zeros((h * rows, w * cols, c))

        for i, x in enumerate(a):
            row = i // cols
            col = i % cols
            new_a[row * h:(row + 1) * h, col * w:(col + 1) * w] = x

        return np.squeeze(new_a)


class Visualizer(object):
    """
    Visualizes attributions directly as a color image. Intended particularly for
    use with input-attributions.

    This can also be used for viewing images (rather than attributions).
    """

    def __init__(
        self,
        combine_channels: bool = False,
        normalization_type: str = None,
        blur: float = 0.,
        cmap: 'matplotlib.colors.Colormap' = None
    ):
        """
        Configures the default parameters for the `__call__` method (these can 
        be overridden by passing in values to `__call__`).

        Parameters:
            combine_channels:
                If `True`, the attributions will be averaged across the channel
                dimension, resulting in a 1-channel attribution map.

            normalization_type:
                Specifies one of the following configurations for normalizing
                the attributions (each item is normalized separately):

                - `'unsigned_max'`: normalizes the attributions to the range 
                  [-1, 1] by dividing the attributions by the maximum absolute 
                  attribution value.
                - `'unsigned_max_positive_centered'`: same as above, but scales
                  the values to the range [0, 1], with negative scores less than
                  0.5 and positive scores greater than 0.5. 
                - `'magnitude_max'`: takes the absolute value of the 
                  attributions, then normalizes the attributions to the range 
                  [0, 1] by dividing by the maximum absolute attribution value.
                - `'magnitude_sum'`: takes the absolute value of the 
                  attributions, then scales them such that they sum to 1. If 
                  this option is used, each channel is normalized separately, 
                  such that each channel sums to 1.
                - `'signed_max'`: normalizes the attributions to the range 
                  [-1, 1] by dividing the positive values by the maximum 
                  positive attribution value and the negative values by the 
                  minimum negative attribution value.
                - `'signed_max_positive_centered'`: same as above, but scales 
                  the values to the range [0, 1], with negative scores less than
                  0.5 and positive scores greater than 0.5.
                - `'signed_sum'`: scales the positive attributions such that 
                  they sum to 1 and the negative attributions such that they
                  scale to -1. If this option is used, each channel is 
                  normalized separately.
                - `'01'`: normalizes the attributions to the range [0, 1] by 
                  subtracting the minimum attribution value then dividing by the
                  maximum attribution value.
                - `'unnormalized'`: leaves the attributions unaffected.

                If `None`, either `'unsigned_max'` (for single-channel data) or 
                `'unsigned_max_positive_centered'` (for multi-channel data) is
                used.

            blur:
                Gives the radius of a Gaussian blur to be applied to the 
                attributions before visualizing. This can be used to help focus
                on salient regions rather than specific salient pixels.

            cmap: matplotlib.colors.Colormap | str, optional
                Colormap or name of a Colormap to use for the visualization. If 
                `None`, the colormap will be chosen based on the normalization 
                type. This argument is only used for single-channel data
                (including when `combine_channels` is True).
        """
        purpose = "Image attribution visualization"

        self.m_plt = try_import("matplotlib.pyplot", purpose)
        self.m_cm = try_import("matplotlib", purpose)
        self.m_colors = try_import("matplotlib.colors", purpose)
        self.m_filters = try_import("scipy.ndimage.filters", purpose)

        self.default_combine_channels = combine_channels
        self.default_normalization_type = normalization_type
        self.default_blur = blur
        self.default_cmap = cmap if cmap is not None else self._get_hotcold()

        # TODO(klas): in the future we can allow configuring of tiling settings
        #   by allowing the user to specify the tiler.
        self.tiler = Tiler()

    def __call__(
        self,
        attributions,
        output_file=None,
        imshow=True,
        fig=None,
        return_tiled=False,
        combine_channels=None,
        normalization_type=None,
        blur=None,
        cmap=None
    ) -> np.ndarray:
        """
        Visualizes the given attributions.

        Parameters:
            attributions:
                A `np.ndarray` containing the attributions to be visualized.

            output_file:
                File name to save the visualization image to. If `None`, no
                image will be saved, but the figure can still be displayed.

            imshow:
                If true, a the visualization will be displayed. Otherwise the
                figure will not be displayed, but the figure can still be saved.

            fig:
                The `pyplot` figure to display the visualization in. If `None`,
                a new figure will be created.

            return_tiled:
                If true, the returned array will be in the same shape as the
                visualization, with no batch dimension and the samples in the
                batch tiled along the width and height dimensions. If false, the
                returned array will be reshaped to match `attributions`.

            combine_channels:
                If `True`, the attributions will be averaged across the channel
                dimension, resulting in a 1-channel attribution map. If `None`,
                defaults to the value supplied to the constructor.

            normalization_type:
                Specifies one of the following configurations for normalizing
                the attributions (each item is normalized separately):

                - `'unsigned_max'`: normalizes the attributions to the range 
                  [-1, 1] by dividing the attributions by the maximum absolute 
                  attribution value.
                - `'unsigned_max_positive_centered'`: same as above, but scales
                  the values to the range [0, 1], with negative scores less than
                  0.5 and positive scores greater than 0.5. 
                - `'magnitude_max'`: takes the absolute value of the 
                  attributions, then normalizes the attributions to the range 
                  [0, 1] by dividing by the maximum absolute attribution value.
                - `'magnitude_sum'`: takes the absolute value of the 
                  attributions, then scales them such that they sum to 1. If 
                  this option is used, each channel is normalized separately, 
                  such that each channel sums to 1.
                - `'signed_max'`: normalizes the attributions to the range 
                  [-1, 1] by dividing the positive values by the maximum 
                  positive attribution value and the negative values by the 
                  minimum negative attribution value.
                - `'signed_max_positive_centered'`: same as above, but scales 
                  the values to the range [0, 1], with negative scores less than
                  0.5 and positive scores greater than 0.5.
                - `'signed_sum'`: scales the positive attributions such that 
                  they sum to 1 and the negative attributions such that they
                  scale to -1. If this option is used, each channel is 
                  normalized separately.
                - `'01'`: normalizes the attributions to the range [0, 1] by 
                  subtracting the minimum attribution value then dividing by the
                  maximum attribution value.
                - `'unnormalized'`: leaves the attributions unaffected.

                If `None`, defaults to the value supplied to the constructor.

            blur:
                Gives the radius of a Gaussian blur to be applied to the 
                attributions before visualizing. This can be used to help focus
                on salient regions rather than specific salient pixels. If
                `None`, defaults to the value supplied to the constructor.

            cmap: matplotlib.colors.Colormap | str, optional
                Colormap or name of a Colormap to use for the visualization. If
                `None`, defaults to the value supplied to the constructor.

        Returns:
            A `np.ndarray` array of the numerical representation of the
            attributions as modified for the visualization. This includes 
            normalization, blurring, etc.
        """
        combine_channels, normalization_type, blur, cmap = self._check_args(
            attributions, combine_channels, normalization_type, blur, cmap
        )

        # Combine the channels if specified.
        if combine_channels:
            attributions = attributions.mean(
                axis=get_backend().channel_axis, keepdims=True
            )

        # Blur the attributions so the explanation is smoother.
        if blur:
            attributions = self._blur(attributions, blur)

        # Normalize the attributions.
        attributions = self._normalize(attributions, normalization_type)

        tiled_attributions = self.tiler.tile(attributions)

        # Display the figure:
        _fig = self.m_plt.figure() if fig is None else fig

        self.m_plt.axis('off')
        self.m_plt.imshow(tiled_attributions, cmap=cmap)

        if output_file:
            self.m_plt.savefig(output_file, bbox_inches=0)

        if imshow:
            self.m_plt.show()

        elif fig is None:
            self.m_plt.close(_fig)

        return tiled_attributions if return_tiled else attributions

    def _check_args(
        self, attributions, combine_channels, normalization_type, blur, cmap
    ):
        """
        Validates the arguments, and sets them to their default values if they
        are not specified.
        """
        if attributions.ndim != 4:
            raise ValueError(
                '`Visualizer` is inteded for 4-D image-format data. Given '
                'input with dimension {}'.format(attributions.ndim)
            )

        if combine_channels is None:
            combine_channels = self.default_combine_channels

        channel_axis = get_backend().channel_axis
        if not (attributions.shape[channel_axis] in (1, 3, 4) or
                combine_channels):

            raise ValueError(
                'To visualize, attributions must have either 1, 3, or 4 color '
                'channels, but `Visualizer` got {} channels.\n'
                'If you are visualizing an internal layer, consider setting '
                '`combine_channels` to True'.format(
                    attributions.shape[channel_axis]
                )
            )

        if normalization_type is None:
            normalization_type = self.default_normalization_type

            if normalization_type is None:
                if combine_channels or attributions.shape[channel_axis] == 1:
                    normalization_type = 'unsigned_max'

                else:
                    normalization_type = 'unsigned_max_positive_centered'

        valid_normalization_types = [
            'unsigned_max',
            'unsigned_max_positive_centered',
            'magnitude_max',
            'magnitude_sum',
            'signed_max',
            'signed_max_positive_centered',
            'signed_sum',
            '01',
            'unnormalized',
        ]
        if normalization_type not in valid_normalization_types:
            raise ValueError(
                '`norm` must be None or one of the following options:' +
                ','.join(
                    [
                        '\'{}\''.form(norm_type)
                        for norm_type in valid_normalization_types
                    ]
                )
            )

        if blur is None:
            blur = self.default_blur

        if cmap is None:
            cmap = self.default_cmap

        return combine_channels, normalization_type, blur, cmap

    def _normalize(self, attributions, normalization_type, eps=1e-20):
        channel_axis = get_backend().channel_axis
        if normalization_type == 'unnormalized':
            return attributions

        split_by_channel = normalization_type.endswith('sum')

        channel_split = [attributions] if split_by_channel else np.split(
            attributions, attributions.shape[channel_axis], axis=channel_axis
        )

        normalized_attributions = []
        for c_map in channel_split:
            if normalization_type == 'magnitude_max':
                c_map = np.abs(c_map) / (
                    np.abs(c_map).max(axis=(1, 2, 3), keepdims=True) + eps
                )

            elif normalization_type == 'magnitude_sum':
                c_map = np.abs(c_map) / (
                    np.abs(c_map).sum(axis=(1, 2, 3), keepdims=True) + eps
                )

            elif normalization_type.startswith('signed_max'):
                postive_max = c_map.max(axis=(1, 2, 3), keepdims=True)
                negative_max = (-c_map).max(axis=(1, 2, 3), keepdims=True)

                # Normalize the postive socres to [0, 1] and negative socresn to
                # [-1, 0].
                normalization_factor = np.where(
                    c_map >= 0, postive_max, negative_max
                )
                c_map = c_map / (normalization_factor + eps)

                # If positive-centered, normalize so that all scores are in the
                # range [0, 1], with negative scores less than 0.5 and positive
                # scores greater than 0.5.
                if normalization_type.endswith('positive_centered'):
                    c_map = c_map / 2. + 0.5

            elif normalization_type == 'signed_sum':
                postive_max = np.maximum(c_map, 0).sum(
                    axis=(1, 2, 3), keepdims=True
                )
                negative_max = np.maximum(-c_map, 0).sum(
                    axis=(1, 2, 3), keepdims=True
                )

                # Normalize the postive socres to ensure they sum to 1 and the
                # negative scores to ensure they sum to -1.
                normalization_factor = np.where(
                    c_map >= 0, postive_max, negative_max
                )
                c_map = c_map / (normalization_factor + eps)

            elif normalization_type.startswith('unsigned_max'):
                c_map = c_map / (
                    np.abs(c_map).max(axis=(1, 2, 3), keepdims=True) + eps
                )

                # If positive-centered, normalize so that all scores are in the
                # range [0, 1], with negative scores less than 0.5 and positive
                # scores greater than 0.5.
                if normalization_type.endswith('positive_centered'):
                    c_map = c_map / 2. + 0.5

            elif normalization_type == '01':
                c_map = c_map - c_map.min(axis=(1, 2, 3), keepdims=True)
                c_map = c_map / (c_map.max(axis=(1, 2, 3), keepdims=True) + eps)

            normalized_attributions.append(c_map)

        return np.concatenate(normalized_attributions, axis=channel_axis)

    def _blur(self, attributions, blur):
        for i in range(attributions.shape[0]):
            attributions[i] = self.m_filters.gaussian_filter(
                attributions[i], blur
            )

        return attributions

    def _get_hotcold(self):
        hot = self.m_cm.get_cmap('hot', 128)
        cool = self.m_cm.get_cmap('cool', 128)
        binary = self.m_cm.get_cmap('binary', 128)
        hotcold = np.vstack(
            (
                binary(np.linspace(0, 1, 128)) * cool(np.linspace(0, 1, 128)),
                hot(np.linspace(0, 1, 128))
            )
        )

        return self.m_colors.ListedColormap(hotcold, name='hotcold')


class HeatmapVisualizer(Visualizer):
    """
    Visualizes attributions by overlaying an attribution heatmap over the
    original image, similar to how GradCAM visualizes attributions.
    """

    def __init__(
        self,
        overlay_opacity=0.5,
        normalization_type=None,
        blur=10.,
        cmap='jet'
    ):
        """
        Configures the default parameters for the `__call__` method (these can 
        be overridden by passing in values to `__call__`).

        Parameters:
            overlay_opacity: float
                Value in the range [0, 1] specifying the opacity for the heatmap
                overlay.

            normalization_type:
                Specifies one of the following configurations for normalizing
                the attributions (each item is normalized separately):

                - `'unsigned_max'`: normalizes the attributions to the range 
                  [-1, 1] by dividing the attributions by the maximum absolute 
                  attribution value.
                - `'unsigned_max_positive_centered'`: same as above, but scales
                  the values to the range [0, 1], with negative scores less than
                  0.5 and positive scores greater than 0.5. 
                - `'magnitude_max'`: takes the absolute value of the 
                  attributions, then normalizes the attributions to the range 
                  [0, 1] by dividing by the maximum absolute attribution value.
                - `'magnitude_sum'`: takes the absolute value of the 
                  attributions, then scales them such that they sum to 1. If 
                  this option is used, each channel is normalized separately, 
                  such that each channel sums to 1.
                - `'signed_max'`: normalizes the attributions to the range 
                  [-1, 1] by dividing the positive values by the maximum 
                  positive attribution value and the negative values by the 
                  minimum negative attribution value.
                - `'signed_max_positive_centered'`: same as above, but scales 
                  the values to the range [0, 1], with negative scores less than
                  0.5 and positive scores greater than 0.5.
                - `'signed_sum'`: scales the positive attributions such that 
                  they sum to 1 and the negative attributions such that they
                  scale to -1. If this option is used, each channel is 
                  normalized separately.
                - `'01'`: normalizes the attributions to the range [0, 1] by 
                  subtracting the minimum attribution value then dividing by the
                  maximum attribution value.
                - `'unnormalized'`: leaves the attributions unaffected.

                If `None`, either `'unsigned_max'` (for single-channel data) or 
                `'unsigned_max_positive_centered'` (for multi-channel data) is
                used.

            blur:
                Gives the radius of a Gaussian blur to be applied to the 
                attributions before visualizing. This can be used to help focus
                on salient regions rather than specific salient pixels.

            cmap: matplotlib.colors.Colormap | str, optional
                Colormap or name of a Colormap to use for the visualization. If 
                `None`, the colormap will be chosen based on the normalization 
                type. This argument is only used for single-channel data
                (including when `combine_channels` is True).
        """

        super().__init__(
            combine_channels=True,
            normalization_type=normalization_type,
            blur=blur,
            cmap=cmap
        )

        self.default_overlay_opacity = overlay_opacity

    def __call__(
        self,
        attributions,
        x,
        output_file=None,
        imshow=True,
        fig=None,
        return_tiled=False,
        overlay_opacity=None,
        normalization_type=None,
        blur=None,
        cmap=None
    ) -> np.ndarray:
        """
        Visualizes the given attributions by overlaying an attribution heatmap 
        over the given image.

        Parameters:
            attributions:
                A `np.ndarray` containing the attributions to be visualized.

            x:
                A `np.ndarray` of items in the same shape as `attributions`
                corresponding to the records explained by the given 
                attributions. The visualization will be superimposed onto the
                corresponding set of records.

            output_file:
                File name to save the visualization image to. If `None`, no
                image will be saved, but the figure can still be displayed.

            imshow:
                If true, a the visualization will be displayed. Otherwise the
                figure will not be displayed, but the figure can still be saved.

            fig:
                The `pyplot` figure to display the visualization in. If `None`,
                a new figure will be created.

            return_tiled:
                If true, the returned array will be in the same shape as the
                visualization, with no batch dimension and the samples in the
                batch tiled along the width and height dimensions. If false, the
                returned array will be reshaped to match `attributions`.

            overlay_opacity: float
                Value in the range [0, 1] specifying the opacity for the heatmap
                overlay. If `None`, defaults to the value supplied to the 
                constructor.

            normalization_type:
                Specifies one of the following configurations for normalizing
                the attributions (each item is normalized separately):

                - `'unsigned_max'`: normalizes the attributions to the range 
                  [-1, 1] by dividing the attributions by the maximum absolute 
                  attribution value.
                - `'unsigned_max_positive_centered'`: same as above, but scales
                  the values to the range [0, 1], with negative scores less than
                  0.5 and positive scores greater than 0.5. 
                - `'magnitude_max'`: takes the absolute value of the 
                  attributions, then normalizes the attributions to the range 
                  [0, 1] by dividing by the maximum absolute attribution value.
                - `'magnitude_sum'`: takes the absolute value of the 
                  attributions, then scales them such that they sum to 1. If 
                  this option is used, each channel is normalized separately, 
                  such that each channel sums to 1.
                - `'signed_max'`: normalizes the attributions to the range 
                  [-1, 1] by dividing the positive values by the maximum 
                  positive attribution value and the negative values by the 
                  minimum negative attribution value.
                - `'signed_max_positive_centered'`: same as above, but scales 
                  the values to the range [0, 1], with negative scores less than
                  0.5 and positive scores greater than 0.5.
                - `'signed_sum'`: scales the positive attributions such that 
                  they sum to 1 and the negative attributions such that they
                  scale to -1. If this option is used, each channel is 
                  normalized separately.
                - `'01'`: normalizes the attributions to the range [0, 1] by 
                  subtracting the minimum attribution value then dividing by the
                  maximum attribution value.
                - `'unnormalized'`: leaves the attributions unaffected.

                If `None`, defaults to the value supplied to the constructor.

            blur:
                Gives the radius of a Gaussian blur to be applied to the 
                attributions before visualizing. This can be used to help focus
                on salient regions rather than specific salient pixels. If
                `None`, defaults to the value supplied to the constructor.

            cmap: matplotlib.colors.Colormap | str, optional
                Colormap or name of a Colormap to use for the visualization. If
                `None`, defaults to the value supplied to the constructor.

        Returns:
            A `np.ndarray` array of the numerical representation of the
            attributions as modified for the visualization. This includes 
            normalization, blurring, etc.
        """
        _, normalization_type, blur, cmap = self._check_args(
            attributions, None, normalization_type, blur, cmap
        )

        # Combine the channels.
        attributions = attributions.mean(
            axis=get_backend().channel_axis, keepdims=True
        )

        # Blur the attributions so the explanation is smoother.
        if blur:
            attributions = self._blur(attributions, blur)

        # Normalize the attributions.
        attributions = self._normalize(attributions, normalization_type)

        tiled_attributions = self.tiler.tile(attributions)

        # Normalize the pixels to be in the range [0, 1].
        x = self._normalize(x, '01')
        tiled_x = self.tiler.tile(x)

        if cmap is None:
            cmap = self.default_cmap

        if overlay_opacity is None:
            overlay_opacity = self.default_overlay_opacity

        # Display the figure:
        _fig = self.m_plt.figure() if fig is None else fig

        self.m_plt.axis('off')
        self.m_plt.imshow(tiled_x)
        self.m_plt.imshow(tiled_attributions, alpha=overlay_opacity, cmap=cmap)

        if output_file:
            self.m_plt.savefig(output_file, bbox_inches=0)

        if imshow:
            self.m_plt.show()

        elif fig is None:
            self.m_plt.close(_fig)

        return tiled_attributions if return_tiled else attributions


class MaskVisualizer(object):
    """
    Visualizes attributions by masking the original image to highlight the
    regions with influence above a given threshold percentile. Intended 
    particularly for use with input-attributions.
    """

    def __init__(
        self,
        blur=5.,
        threshold=0.5,
        masked_opacity=0.2,
        combine_channels=True,
        use_attr_as_opacity=False,
        positive_only=True
    ):
        """
        Configures the default parameters for the `__call__` method (these can 
        be overridden by passing in values to `__call__`).

        Parameters:
            blur:
                Gives the radius of a Gaussian blur to be applied to the 
                attributions before visualizing. This can be used to help focus
                on salient regions rather than specific salient pixels.

            threshold:
                Value in the range [0, 1]. Attribution values at or  below the 
                percentile given by `threshold` (after normalization, blurring,
                etc.) will be masked.

            masked_opacity: 
                Value in the range [0, 1] specifying the opacity for the parts
                of the image that are masked.

            combine_channels:
                If `True`, the attributions will be averaged across the channel
                dimension, resulting in a 1-channel attribution map.

            use_attr_as_opacity:
                If `True`, instead of using `threshold` and `masked_opacity`,
                the opacity of each pixel is given by the 0-1-normalized 
                attribution value.

            positive_only:
                If `True`, only pixels with positive attribution will be 
                unmasked (or given nonzero opacity when `use_attr_as_opacity` is
                true).
        """

        # TODO: Figure out why this is not a subclass of Visualizer.

        purpose = "Mask visualization"

        self.m_plt = try_import("matplotlib.pyplot", purpose)
        self.m_cm = try_import("matplotlib", purpose)
        self.m_colors = try_import("matplotlib.colors", purpose)
        self.m_filters = try_import("scipy.ndimage.filters", purpose)

        self.default_blur = blur
        self.default_thresh = threshold
        self.default_masked_opacity = masked_opacity
        self.default_combine_channels = combine_channels

        # TODO(klas): in the future we can allow configuring of tiling settings
        #   by allowing the user to specify the tiler.
        self.tiler = Tiler()

    def __call__(
        self,
        attributions,
        x,
        output_file=None,
        imshow=True,
        fig=None,
        return_tiled=True,
        blur=None,
        threshold=None,
        masked_opacity=None,
        combine_channels=None,
        use_attr_as_opacity=None,
        positive_only=None
    ):
        channel_axis = get_backend().channel_axis
        if attributions.shape != x.shape:
            raise ValueError(
                'Shape of `attributions` {} must match shape of `x` {}'.format(
                    attributions.shape, x.shape
                )
            )

        if blur is None:
            blur = self.default_blur

        if threshold is None:
            threshold = self.default_thresh

        if masked_opacity is None:
            masked_opacity = self.default_masked_opacity

        if combine_channels is None:
            combine_channels = self.default_combine_channels

        if len(attributions.shape) != 4:
            raise ValueError(
                '`MaskVisualizer` is inteded for 4-D image-format data. Given '
                'input with dimension {}'.format(len(attributions.shape))
            )

        if combine_channels is None:
            combine_channels = self.default_combine_channels

        if combine_channels:
            attributions = attributions.mean(axis=channel_axis, keepdims=True)

        if x.shape[channel_axis] not in (1, 3, 4):
            raise ValueError(
                'To visualize, attributions must have either 1, 3, or 4 color '
                'channels, but Visualizer got {} channels.\n'
                'If you are visualizing an internal layer, consider setting '
                '`combine_channels` to True'.format(
                    attributions.shape[channel_axis]
                )
            )

        # Blur the attributions so the explanation is smoother.
        if blur is not None:
            attributions = [
                self.m_filters.gaussian_filter(a, blur) for a in attributions
            ]

        # If `positive_only` clip attributions.
        if positive_only:
            attributions = np.maximum(attributions, 0)

        # Normalize the attributions to be in the range [0, 1].
        attributions = [a - a.min() for a in attributions]
        attributions = [
            0. * a if a.max() == 0. else a / a.max() for a in attributions
        ]

        # Normalize the pixels to be in the range [0, 1]
        x = [xc - xc.min() for xc in x]
        x = np.array([0. * xc if xc.max() == 0. else xc / xc.max() for xc in x])

        # Threshold the attributions to create a mask.
        if threshold is not None:
            percentiles = [
                np.percentile(a, 100 * threshold) for a in attributions
            ]
            masks = np.array(
                [
                    np.maximum(a > p, masked_opacity)
                    for a, p in zip(attributions, percentiles)
                ]
            )

        else:
            masks = np.array(attributions)

        # Use the mask on the original image to visualize the explanation.
        attributions = masks * x
        tiled_attributions = self.tiler.tile(attributions)

        if imshow:
            self.m_plt.axis('off')
            self.m_plt.imshow(tiled_attributions)

            if output_file:
                self.m_plt.savefig(output_file, bbox_inches=0)

        return tiled_attributions if return_tiled else attributions


class ChannelMaskVisualizer(object):
    """
    Uses internal influence to visualize the pixels that are most salient
    towards a particular internal channel or neuron.
    """

    def __init__(
        self,
        model,
        layer,
        channel,
        channel_axis=None,
        agg_fn=None,
        doi=None,
        blur=None,
        threshold=0.5,
        masked_opacity=0.2,
        combine_channels: bool = True,
        use_attr_as_opacity=None,
        positive_only=None
    ):
        """
        Configures the default parameters for the `__call__` method (these can 
        be overridden by passing in values to `__call__`).

        Parameters:
            model:
                The wrapped model whose channel we're visualizing.

            layer:
                The identifier (either index or name) of the layer in which the 
                channel we're visualizing resides.

            channel:
                Index of the channel (for convolutional layers) or internal 
                neuron (for fully-connected layers) that we'd like to visualize.

            channel_axis:
                If different from the channel axis specified by the backend, the
                supplied `channel_axis` will be used if operating on a 
                convolutional layer with 4-D image format.

            agg_fn:
                Function with which to aggregate the remaining dimensions 
                (except the batch dimension) in order to get a single scalar 
                value for each channel; If `None`, a sum over each neuron in the
                channel will be taken. This argument is not used when the 
                channels are scalars, e.g., for dense layers.

            doi:
                The distribution of interest to use when computing the input
                attributions towards the specified channel. If `None`, 
                `PointDoI` will be used.

            blur:
                Gives the radius of a Gaussian blur to be applied to the 
                attributions before visualizing. This can be used to help focus
                on salient regions rather than specific salient pixels.

            threshold:
                Value in the range [0, 1]. Attribution values at or  below the 
                percentile given by `threshold` (after normalization, blurring,
                etc.) will be masked.

            masked_opacity: 
                Value in the range [0, 1] specifying the opacity for the parts
                of the image that are masked.

            combine_channels:
                If `True`, the attributions will be averaged across the channel
                dimension, resulting in a 1-channel attribution map.

            use_attr_as_opacity:
                If `True`, instead of using `threshold` and `masked_opacity`,
                the opacity of each pixel is given by the 0-1-normalized 
                attribution value.

            positive_only:
                If `True`, only pixels with positive attribution will be 
                unmasked (or given nonzero opacity when `use_attr_as_opacity` is
                true).
        """

        # TODO: Figure out why this is not a subclass of Visualizer.

        purpose = "Channel mask visualization"

        self.m_plt = try_import("matplotlib.pyplot", purpose)
        self.m_cm = try_import("matplotlib", purpose)
        self.m_colors = try_import("matplotlib.colors", purpose)
        self.m_filters = try_import("scipy.ndimage.filters", purpose)

        B = get_backend()
        if (B is not None and (channel_axis is None or channel_axis < 0)):
            channel_axis = B.channel_axis
        elif (channel_axis is None or channel_axis < 0):
            channel_axis = 1

        self.mask_visualizer = MaskVisualizer(
            blur, threshold, masked_opacity, combine_channels,
            use_attr_as_opacity, positive_only
        )

        self.infl_input = InternalInfluence(
            model, (InputCut(), Cut(layer)),
            InternalChannelQoI(channel, channel_axis, agg_fn),
            PointDoi() if doi is None else doi
        )

    def __call__(
        self,
        x,
        x_preprocessed=None,
        output_file=None,
        blur=None,
        threshold=None,
        masked_opacity=None,
        combine_channels=None
    ):
        """
        Visualizes the given attributions by overlaying an attribution heatmap 
        over the given image.

        Parameters
        ----------
        attributions : numpy.ndarray
            The attributions to visualize. Expected to be in 4-D image format.

        x : numpy.ndarray
            The original image(s) over which the attributions are calculated.
            Must be the same shape as expected by the model used with this
            visualizer.

        x_preprocessed : numpy.ndarray, optional
            If the model requires a preprocessed input (e.g., with the mean
            subtracted) that is different from how the image should be 
            visualized, ``x_preprocessed`` should be specified. In this case 
            ``x`` will be used for visualization, and ``x_preprocessed`` will be
            passed to the model when calculating attributions. Must be the same 
            shape as ``x``.

        output_file : str, optional
            If specified, the resulting visualization will be saved to a file
            with the name given by ``output_file``.

        blur : float, optional
            If specified, gives the radius of a Gaussian blur to be applied to
            the attributions before visualizing. This can be used to help focus
            on salient regions rather than specific salient pixels. If None, 
            defaults to the value supplied to the constructor. Default None.

        threshold : float
            Value in the range [0, 1]. Attribution values at or  below the 
            percentile given by ``threshold`` will be masked. If None, defaults 
            to the value supplied to the constructor. Default None.

        masked_opacity: float
            Value in the range [0, 1] specifying the opacity for the parts of
            the image that are masked. Default 0.2. If None, defaults to the 
            value supplied to the constructor. Default None.

        combine_channels : bool
            If True, the attributions will be averaged across the channel
            dimension, resulting in a 1-channel attribution map. If None, 
            defaults to the value supplied to the constructor. Default None.
        """

        attrs_input = self.infl_input.attributions(
            x if x_preprocessed is None else x_preprocessed
        )

        return self.mask_visualizer(
            attrs_input, x, output_file, blur, threshold, masked_opacity,
            combine_channels
        )


# TODO: Unify visualization parameters for vision above and for nlp below.

# A colormap is a method that given a value between -1.0 and 1.0, returns a quad
# of rgba values, each floating between 0.0 and 1.0.
RGBA = Tuple[float, float, float, float]
COLORMAP = Callable[[float], RGBA]


class ColorMap:

    @staticmethod
    def of_matplotlib(divergent=None, positive=None, negative=None):
        """Convert a matplotlib color map which expects values from [0.0, 1.0] into one we expect with values in [-1.0, 1.0]."""

        if divergent is None:
            if positive is None or negative is None:
                raise ValueError(
                    "To convert a matplotlib colormap, provide either a symmetric divergent parameter or both positive and negative parameters."
                )

            return lambda f: positive(f) if f >= 0.0 else negative(-f)
        else:
            if positive is not None or negative is not None:
                raise ValueError(
                    "To convert a matplotlib colormap, provide either a symmetric divergent parameter or both positive and negative parameters."
                )

            return lambda f: divergent((f + 1.0) / 2.0)

    @staticmethod
    def default(f: float) -> RGBA:  # :COLORMAP
        if f > 1.0:
            f = 1.0
        if f < -1.0:
            f = -1.0

        red = 0.0
        green = 0.0
        if f > 0:
            green = 1.0  # 0.5 + mag * 0.5
            red = 1.0 - f
        else:
            red = 1.0
            green = 1.0 + f
            #red = 0.5 - mag * 0.5

        blue = min(red, green)
        # blue = 1.0 - max(red, green)

        return (red, green, blue, 1.0)


def arrays_different(a: np.ndarray, b: np.ndarray):
    """
    Given two arrays of potentially different lengths, return a boolean array
    indicating in which indices the two are different. If one is shorter than
    the other, the indices past the end of the shorter array are marked as
    different.
    """

    m = min(len(a), len(b))
    M = max(len(a), len(b))

    diff = np.array([True] * M)
    diff[0:m] = a[0:m] != b[0:m]

    return diff


class Output(ABC):
    """Base class for visualization output formats."""

    # Element type
    E = TypeVar("E")

    # Rendered output type
    R = TypeVar("R")

    @abstractmethod
    def blank(self) -> E:
        ...

    @abstractmethod
    def space(self) -> E:
        ...

    @abstractmethod
    def big(self, s: E) -> E:
        ...

    @abstractmethod
    def highlight(self, s: E) -> E:
        ...

    @abstractmethod
    def sub(self, e: E, color: str) -> E:
        ...

    def scores(
        self, scores: np.ndarray, labels: List[str], qoi: QoI = None
    ) -> E:
        if sum(scores) != 1.0:
            scores = np.exp(scores) / np.exp(scores).sum()

        highlights = [False] * len(scores)

        color = "rgba(0, 100, 100, 1.0)"

        content = []

        pred = np.argmax(scores)

        if isinstance(qoi, ClassQoI):
            highlights[qoi.cl] = color
            content += [
                self.label(f"ClassQoI", color=color),
                self.sub(str(qoi.cl), color=color),
                self.space()
            ]

        if isinstance(qoi, MaxClassQoI):
            highlights = [color] * len(scores)
            content += [self.label(f"MaxClassQoI", color=color), self.space()]

        if isinstance(qoi, ComparativeQoI):
            highlights[qoi.cl1] = "green"
            highlights[qoi.cl2] = "red"
            content += [
                self.label(f"ComparativeQoI", color=color),
                self.sub(str(qoi.cl1), color="green"),
                self.space(),
                self.sub(str(qoi.cl2), color="red"),
                self.space()
            ]

        for i, (score, label, highlight) in enumerate(zip(scores, labels,
                                                          highlights)):
            temp = self.magnitude_colored(label, mag=score)

            if highlight is not None:
                temp = self.highlight(temp, color=highlight)

            if pred == i:
                temp = self.highlight(temp, color="white")

            content += [temp]

            if i + 1 < len(scores):
                content += [self.space()]

        #content += [self.label(f")")]

        return self.concat(*content)

    @abstractmethod
    def token(self, s: str, token_id=None) -> E:
        ...

    @abstractmethod
    def label(self, s: str) -> E:
        ...

    @abstractmethod
    def line(self, e: E) -> E:
        ...

    @abstractmethod
    def magnitude_colored(self, s: str, mag: float, color_map: COLORMAP) -> E:
        ...

    @abstractmethod
    def concat(self, *parts: Iterable[E]) -> E:
        ...

    @abstractmethod
    def render(self, e: E) -> R:
        ...

    @abstractmethod
    def open(self, r: R) -> None:
        ...


# TODO(piotrm): implement a latex output format


class EnvType:
    ...


class Term(EnvType):
    ...


class Jupyter(EnvType):
    ...


class Colab(Jupyter):
    ...


def guess_env_type():
    # From Andreas
    '''
    Tests whether current process is running in a:
            o terminal as a regular Python shell
            o jupyter notebook
            o Google colab
    returns one of {'terminal', 'jupyter', 'colab', None}
    None means could not determine.
    '''
    try:
        from IPython import get_ipython
    except ImportError:
        return Term()
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return Jupyter()  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return Term()  # Terminal running IPython
        elif shell == 'Shell' and get_ipython(
        ).__class__.__module__ == 'google.colab._shell':
            return Colab()
        else:
            return Term()  # Other type (?)
    except NameError:
        return Term()  # Probably standard Python interpreter


class PlainText(Output):
    """Plain text visualization output format."""

    E = str
    R = str

    def blank(self) -> E:
        return ""

    def space(self) -> E:
        return " "

    def big(self, s: E) -> E:
        return f"_{s}_"

    def highlight(self, s: E, color: str) -> E:
        return f"[{s}]"

    def sub(self, s: E, color: str) -> E:
        return f".{s}."

    def token(self, s: str, token_id=None) -> E:
        s = self.label(s)

        content = s

        if token_id is not None:
            content += f"({token_id})"

        return content

    def label(self, s: str, color: str) -> E:
        return s

    def line(self, e: E) -> E:
        return e

    def magnitude_colored(
        self, s: str, mag: float, color_map: COLORMAP = ColorMap.default
    ) -> E:
        return f"{self.label(s)}({mag:0.3f})"

    def concat(self, *parts: Iterable[E]) -> E:
        return ''.join(parts)

    def render(self, e: E) -> R:
        return e

    def open(self, r: R) -> None:
        raise NotImplementedError


class HTML(Output):
    """HTML visualization output format."""

    E = 'domonic.dom.Node'
    R = str

    def __init__(self):
        self.m_html_util = try_import("html", msg="html output")
        self.m_dom = try_import("domonic.dom", msg="html output")
        self.m_html = try_import("domonic", msg="html output")

    def blank(self) -> E:
        return self.m_dom.Document.createDocumentFragment()

    def space(self) -> E:
        return self.m_dom.Text("&nbsp;")

    def label(self, s: str, color: str = "white") -> E:
        return self.m_html.span(
            self.m_dom.Text(self.m_html_util.escape(s)),
            style=f"color: {color};"
        )

    def linebreak(self) -> E:
        return self.m_html.br()

    def line(self, e: E) -> E:
        return self.m_html.div(
            e, style="padding: 5px; maring: 0px; background: black;"
        )

    def big(self, e: E) -> E:
        return self.m_html.strong(e)

    def highlight(self, s: E, color: str = "white") -> E:
        return self.m_html.span(
            s, style=f"padding: 2px; border: 2px solid {color};"
        )

    def sub(self, e: E, color: str = "white") -> E:
        return self.m_html.sub(e, style=f"color: {color};")

    def token(self, s: str, token_id=None) -> E:
        s = self.label(s)

        extra_arg = {}
        if token_id is not None:
            extra_arg['title'] = f"token id: {token_id}"

        pad_top = 0
        pad_bot = 2

        return self.m_html.span(
            s,
            style=
            f'border-top: {pad_top}px solid gray; border-bottom: {pad_bot}px solid gray; margin-left 1px; margin-right: 1px; background: black; color: white;',
            **extra_arg
        )

    def scale(self, mag: float, color_map=ColorMap.default) -> E:

        if isinstance(mag, tuple):
            maglabel = f"{mag[1]:0.1f}{mag[0]:0.1f}"
        else:
            maglabel = f"{mag:0.3f}"

        if isinstance(mag, tuple):
            magn = mag[0]
            magp = mag[1]
            rn, gn, bn, an = np.array(color_map(mag[0])) * 255
            rp, gp, bp, ap = np.array(color_map(mag[1])) * 255
        else:
            if mag > 0:
                magp = mag
                magn = 0.0
                rp, gp, bp, ap = np.array(color_map(mag)) * 255
                rn, gn, bn, an = 0, 0, 0, 0
            else:
                magp = 0.0
                magn = mag
                rn, gn, bn, an = np.array(color_map(mag)) * 255
                rp, gp, bp, ap = 0, 0, 0, 0

        s = self.label(maglabel)

        rgban = f"rgba({rn}, {gn}, {bn}, {an})"
        rgbap = f"rgba({rp}, {gp}, {bp}, {ap})"

        pad_top = 0
        pad_bot = 0

        pad_top = int(min(magp, 1.0) * 10)
        pad_bot = int(-max(magn, -1.0) * 10)

        pieces = []

        pieces += [
            self.m_html.span(
                "",
                title=f"{maglabel}",
                style=
                f'border-right: {pad_bot}px solid {rgban}; margin-left: {25-pad_bot}px;'
            )
        ]

        pieces += [
            self.m_html.span(
                s,
                title=f"{maglabel}",
                style=
                f'display: inline-block; width: 60px; padding: 2px; text-align: center;',
            )
        ]

        pieces += [
            self.m_html.span(
                "",
                title=f"{maglabel}",
                style=
                f'border-left: {pad_top}px solid {rgbap}; margin-right: {25-pad_top}px;'
            )
        ]

        return self.concat(*pieces)

    MagnitudeLike = Union[float, Tuple[float, float]]

    def magnitude_colored(
        self,
        s: str,
        mag: MagnitudeLike,
        color_map=ColorMap.default,
        color='white'
    ) -> E:
        if isinstance(mag, tuple):
            maglabel = f"{mag[1]:0.2f}{mag[0]:0.2f}"
            magn = mag[0]
            magp = mag[1]
            rn, gn, bn, an = np.array(color_map(mag[0])) * 255
            rp, gp, bp, ap = np.array(color_map(mag[1])) * 255
        else:
            maglabel = f"{mag:0.3f}"
            if mag > 0:
                magp = mag
                magn = 0.0
                rp, gp, bp, ap = np.array(color_map(mag)) * 255
                rn, gn, bn, an = 0, 0, 0, 0
            else:
                magp = 0.0
                magn = mag
                rn, gn, bn, an = np.array(color_map(mag)) * 255
                rp, gp, bp, ap = 0, 0, 0, 0

        s = self.label(s)

        rgban = f"rgba({rn}, {gn}, {bn}, {an})"
        rgbap = f"rgba({rp}, {gp}, {bp}, {ap})"

        pad_top = 0
        pad_bot = 0

        pad_top = int(min(magp, 1.0) * 10)
        pad_bot = int(-max(magn, -1.0) * 10)

        return self.m_html.span(
            s,
            title=f"{maglabel}",
            style=
            f'border-top: {pad_top}px solid {rgbap}; border-bottom: {pad_bot}px solid {rgban}; margin-left 1px; margin-right: 1px; background: black; color={color};'
        )

    def concat(self, *pieces: Iterable[E]) -> E:
        temp = self.blank()
        for piece in pieces:
            temp.appendChild(piece)

        return temp

    def render(self, e: E) -> R:
        return str(self.m_html.html(self.m_html.body(e)))

    def open(self, r):
        mod = try_import("webbrowser", msg="html open")

        # from Andreas

        with tempfile.NamedTemporaryFile(prefix='attrs_', mode='w') as fd:
            fd.write(r)
            mod.open_new_tab(f"file://{fd.name}")


class IPython(HTML):
    """Interactive python visualization output format."""

    def __init__(self):
        super(IPython, self).__init__()

        self.m_ipy = try_import("IPython", "Jupyter output")

    def render(self, e):
        html = HTML.render(self, e)
        return self.m_ipy.display.HTML(html)


@dataclass
class InstanceOptionals:
    # Minimally available when rendering tokens or related.
    texts: str
    input_ids: TensorLike

    # Next two available if model wrapper is known.
    # outputs: TensorLike
    logits: TensorLike

    # Available if attributor is known.
    attributions: Outputs[TensorLike]
    multipliers: TensorLike

    # Available if attributor is known and show_doi is set.
    gradients: Outputs[Uniform[TensorLike]]
    interventions: Uniform[TensorLike]

    # Available if embedder is known.
    embeddings: TensorLike


@dataclass
class InstancesOptionals:
    # Minimally available when rendering tokens or related.
    texts: Inputs[str]
    input_ids: Inputs[TensorLike]

    # Next two available if model wrapper is known.
    # outputs: Inputs[TensorLike] # one for each Input
    logits: Inputs[TensorLike] = None  # one for each Input

    # Available if attributor is known.
    attributions: Inputs[TensorLike] = None
    multipliers: Inputs[TensorLike] = None

    # Available if attributor is known and show_doi is set.
    gradients: Inputs[Uniform[TensorLike]] = None
    interventions: Inputs[Uniform[TensorLike]] = None

    # Available if embedder is known.
    embeddings: Inputs[TensorLike] = None

    def __repr__(self):
        return str(self)

    def __str__(self):
        ret = ""

        for k in dir(self):
            v = getattr(self, k)

            if k not in ["texts", "input_ids", "logits", "attributions",
                         "multipliers", "gradients", "interventions",
                         "embeddings"]:
                continue

            ret += k + "\t: "
            if hasattr(v, "shape"):
                ret += str(v.shape)
            elif hasattr(v, "__len__"):
                ret += f"[{len(v)}]"
            else:
                ret += str(v)

            ret += "\n"

        return ret

    def for_text_index(self, i):
        return InstanceOptionals(
            texts=self.texts[i],
            input_ids=self.input_ids[i],
            # outputs = self.outputs[i],
            logits=self.logits[i] if self.logits is not None else None,
            attributions=self.attributions[i]
            if self.attributions is not None else None,
            multipliers=self.multipliers[i]
            if self.multipliers is not None else None,
            gradients=self.gradients[:,
                                     i] if self.gradients is not None else None,
            interventions=self.interventions[:, i]
            if self.interventions is not None else None,
            embeddings=self.embeddings[i]
            if self.embeddings is not None else None
        )


class NLP(object):
    """NLP Visualization tools."""

    # Batches of text inputs not yet tokenized.
    TextBatch = TypeVar("TextBatch")

    # Inputs that are directly accepted by wrapped models, tokenized.
    # TODO(piotrm): Reuse other typevars/aliases from elsewhere.
    ModelInput = TypeVar("ModelInput")

    # Outputs produced by wrapped models.
    # TODO(piotrm): Reuse other typevars/aliases from elsewhere.
    ModelOutput = TypeVar("ModelOutput")

    def __init__(
        self,
        wrapper: ModelWrapper = None,
        output: Optional[Output] = None,
        labels: Optional[Iterable[str]] = None,
        tokenize: Optional[Callable[[TextBatch], ModelInputs]] = None,
        embedder: Optional[Any] = None,  # fix type hint
        embeddings: Optional[np.ndarray] = None,
        embedding_distance: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        decode: Optional[Callable[[Tensor], str]] = None,
        input_accessor: Optional[Callable[[ModelInputs],
                                          Iterable[Tensor]]] = None,
        output_accessor: Optional[Callable[[ModelOutput],
                                           Iterable[Tensor]]] = None,
        attr_aggregate: Optional[Callable[[Tensor], Tensor]] = None,
        aggregate_twosided: bool = False,
        hidden_tokens: Optional[Set[int]] = set(),
        color_map: Callable[[float], Tuple[float, float, float,
                                           float]] = ColorMap.default
    ):
        """Initializate NLP visualization tools for a given environment.

        Parameters:
            wrapper: ModelWrapper
                The wrapped model whose channel we're visualizing.

            output: Output, optional
                Visualization output format. Defaults to PlainText unless
                ipython is detected and in which case defaults to IPython
                format.

            labels: Iterable[str], optional
                Names of prediction classes for classification models.

            tokenize: Callable[[TextBatch], ModelInput], optional
                Method to tokenize an instance.

            decode: Callable[[Tensor], str], optional
                Method to invert/decode the tokenization.

            input_accessor: Callable[[ModelInputs], Iterable[Tensor]], optional
                Method to extract input/token ids from model inputs (tokenize
                output) if needed.

            output_accessor: Callable[[ModelOutput], Iterable[Tensor]], optional
                Method to extract outout logits from output structures if
                needed.

            attr_aggregate: Callable[[Tensor], Tensor], optional
                Method to aggregate attribution for embedding into a single
                value. Defaults to sum.

            hidden_tokens: Set[int], optional
                For token-based visualizations, which tokens to hide.

            color_map: ColorMap
                Means of coloring floats in [-1.0, 1.0]. 
        """
        if output is None:
            term_type = guess_env_type()
            if isinstance(term_type, Jupyter):
                output = IPython()

            else:
                output = PlainText()
                tru_logger(
                    "WARNING: could not guess preferred visualization output format, using PlainText"
                )

        # TODO: automatic inference of various parameters for common repositories like huggingface, tfhub.

        self.output = output
        self.labels = labels

        self.tokenize = tokenize
        self.decode = decode
        self.wrapper = wrapper

        self.embeddings = embeddings
        self.embedder = embedder
        self.embedding_distance = embedding_distance
        if not isinstance(self.embedding_distance, Callable):
            L = lambda ord: lambda emb: np.linalg.norm(emb - self.embeddings, ord=ord, axis=1)

            if isinstance(self.embedding_distance, str):
                if self.embedding_distance == "l2":
                    self.embedding_distance = L(2.0)
                elif self.embedding_distance == "l1":
                    self.embedding_distance = L(1.0)
                elif self.embedding_distance == "cosine":
                    self.embedding_distance = lambda emb: - np.dot(emb, self.embeddings)
                else:
                    raise ValueError(f"Unknown embedding distance indicator string {self.embedding_distance}")

        self.input_accessor = input_accessor  # could be inferred
        self.output_accessor = output_accessor  # could be inferred

        B = get_backend()

        self.aggregate_twosided = aggregate_twosided

        if attr_aggregate is None:
            if self.aggregate_twosided:
                attr_aggregate = lambda t: (B.sum(t[t < 0]), B.sum(t[t > 0]))
            else:
                attr_aggregate = B.sum

        self.attr_aggregate = attr_aggregate

        self.hidden_tokens = hidden_tokens

        self.color_map = color_map

    def token_attribution_scale(self):
        """
        Render an attribution scale.
        """

        cells = [self.output.label("scale:"), self.output.space()]

        for f in range(-10, 11):
            cells.append(
                self.output.magnitude_colored(
                    str(f / 10.0) if f <= 0 else "+" + str(f / 10.0), f / 10.0,
                    self.color_map
                )
            )

        return self.output.line(self.output.concat(*cells))

    def _tokens_stability_instance(
        self,
        opt: InstanceOptionals,
        show_id=False,
        show_doi=False,
        show_text=False,
        highlights=None,
        attributor=None
    ):
        B = get_backend()

        sent = []

        if self.wrapper is not None:
            logits = B.as_array(opt.logits)
            # TODO: show pred
            # pred = logits.argmax()

            if self.labels is None:
                self.labels = list(map(str, range(0, len(logits))))

            sent += [
                self.output.scores(
                    logits,
                    self.labels,
                    qoi=attributor.qoi if attributor else None
                )
            ]

        if opt.attributions is None:
            attr = [None] * len(opt.input_ids)
        else:
            attr = opt.attributions

        effective_input_ids = [opt.input_ids]
        effective_attr = [attr]

        if opt.embeddings is not None:
            effective_embeddings = [opt.embeddings]
        else:
            effective_embeddings = [None] * len(opt.input_ids)

        if show_doi:
            if opt.interventions is None:
                raise ValueError(
                    "show_doi requires return_doi to be enabled on attributor"
                )
            if opt.gradients is None:
                raise ValueError(
                    "show_doi requires return_grads to be enabled on attributor"
                )

            effective_input_ids += [opt.input_ids] * len(opt.interventions)
            effective_embeddings += list(opt.interventions)
            effective_attr += list(opt.gradients)

        if len(effective_attr) > 1:
            sent += [self.output.linebreak(), self.output.linebreak()]

        for iid, (sentence_input_ids, sentence_embeddings,
                  sentence_attr) in enumerate(zip(effective_input_ids,
                                                  effective_embeddings,
                                                  effective_attr)):

            interv = []

            if show_text and iid == 0:
                interv += [
                    self.output.label(repr(opt.texts)),
                    self.output.space(),
                    self.output.label("=>", color="gray"),
                    self.output.space()
                ]

            if opt.multipliers is None:
                sentence_multiplier = [None] * len(sentence_input_ids)
            else:
                sentence_multiplier = opt.multipliers

            if sentence_embeddings is None:
                sentence_embeddings = [None] * len(sentence_input_ids)

            if iid > 0:
                interv += [
                    self.output.label("+", color="gray"),
                    self.output.space()
                ]

                mag_sentence = self.attr_aggregate(
                    sentence_attr * sentence_multiplier
                )

                interv += [self.output.scale(mag_sentence), self.output.space()]

            for i, (input_id, emb, attr,
                    mult) in enumerate(zip(sentence_input_ids,
                                           sentence_embeddings, sentence_attr,
                                           sentence_multiplier)):

                if show_doi:
                    input_id, dist = self._closest_token(emb)
                else:
                    if input_id in self.hidden_tokens:
                        continue
                    dist = None

                if self.decode is not None:
                    word = self.decode(input_id)
                else:
                    word = str(input_id)

                if word[0] == ' ':
                    word = word[1:]
                    interv += [self.output.space()]

                if word == "":
                    word = "�"

                cap = lambda x: x

                if highlights is not None and highlights[i]:
                    cap = self.output.big

                if attr is not None:
                    if show_doi and iid > 0:  # as in multiply_activations
                        attr = attr * mult  # TODO: need to consider baseline here as well

                    mag = self.attr_aggregate(attr)

                    interv += [
                        cap(
                            self.output.magnitude_colored(
                                word, mag, color_map=self.color_map
                            )
                        )
                    ]
                else:
                    interv += [cap(self.output.token(word, token_id=input_id))]

                if show_id:
                    interv += [
                        self.output.sub(self.output.label(str(input_id)))
                    ]

            if iid == 0 and len(effective_attr) > 1:
                interv += [
                    self.output.space(),
                    self.output.label(
                        f"= (1/{len(opt.interventions)}) * ", color="gray"
                    )
                ]

            sent += self.output.line(self.output.concat(*interv))
            sent += [self.output.linebreak(), self.output.linebreak()]

        return self.output.concat(self.output.line(self.output.concat(*sent)))

    def _get_optionals(
        self, texts: List[str], attributor: AttributionMethod = None
    ):
        B = get_backend()

        def to_numpy(thing):
            return np.array(nested_cast(B, thing, np.ndarray))

        given_inputs = self.tokenize(texts)

        if isinstance(given_inputs, Tensors):
            inputs = given_inputs.as_model_inputs()
        else:
            inputs = ModelInputs(kwargs=given_inputs)

        input_ids = given_inputs
        if self.input_accessor is not None:
            input_ids = self.input_accessor(input_ids)

            if (not isinstance(input_ids, Iterable)) or isinstance(input_ids,
                                                                   dict):
                raise ValueError(
                    f"Inputs ({input_ids.__class__.__name__}) need to be iterable over instances. You might need to set input_accessor."
                )

        # Start constructing the collection of optional fields to return.
        opt = InstancesOptionals(texts=texts, input_ids=to_numpy(input_ids))

        if self.embedder is not None:
            embeddings = self.embedder(input_ids)
            opt.embeddings = to_numpy(embeddings)

        if self.wrapper is not None:
            outputs = inputs.call_on(self.wrapper._model)

            if self.output_accessor is not None:
                opt.logits = self.output_accessor(outputs)
            else:
                opt.logits = outputs

            if (not isinstance(opt.logits, Iterable)) or isinstance(opt.logits,
                                                                    dict):
                raise ValueError(
                    f"Outputs ({opt.logits.__class__.__name__}) need to be iterable over instances. You might need to set output_accessor."
                )

            opt.logits = to_numpy(opt.logits)

        if attributor is not None:
            pieces = attributor._attributions(inputs)

            attributions = pieces.attributions
            if len(attributions) != 1 or len(attributions[0]) != 1:
                raise ValueError(
                    "Only attrubutions with one attribution layer and one qoi output are supported for visualization."
                )

            opt.attributions = to_numpy(attributions)[0, 0]

            if pieces.gradients is not None:
                gradients = pieces.gradients
                opt.gradients = to_numpy(gradients)[0, 0]

            if pieces.interventions is not None:
                interventions = pieces.interventions
                opt.interventions = to_numpy(interventions)[0]

            if self.embedder is not None:
                multipliers = attributor.doi._wrap_public_get_activation_multiplier(
                    activation=embeddings, model_inputs=inputs
                )
                opt.multipliers = to_numpy(multipliers)[0]

        # print(opt)

        return opt

    def tokens_stability(
        self,
        texts1: Iterable[str],
        texts2: Optional[Iterable[str]] = None,
        attributor: Optional[AttributionMethod] = None,
        show_id: bool = False,
        show_doi: bool = False,
        show_scale: bool = True,
        show_text: bool = False
    ):
        """
        Visualize decoded token from sentence pairs. Shows pairs side-by-side
        and highlights differences in them.
        """

        B = get_backend()

        if show_doi:
            if attributor is None:
                raise ValueError(
                    "show_doi requires attributor with doi to be given."
                )

            if not attributor._return_grads or not attributor._return_doi:
                raise ValueError(
                    "show_doi requires attributor to be configured with return_grads, return_doi."
                )

            if self.embedder is None or self.embeddings is None:
                raise ValueError(
                    "show_doi requires embedder and embeddings to be provided to NLP constructor."
                )

        if self.tokenize is None:
            raise ValueError("tokenize not provided to NLP visualizer.")

        textss = [texts1]
        if texts2 is not None:
            textss.append(texts2)

        opts = [
            self._get_optionals(texts, attributor=attributor)
            for texts in textss
        ]

        # Accumulate total output here.
        content = []

        # Include a scale if an attributor was provided.
        if attributor is not None and show_scale:
            content += [
                self.token_attribution_scale(),
                self.output.linebreak(),
                self.output.linebreak()
            ]

        # For each sentence,
        for i in range(len(texts1)):

            opt0 = opts[0].for_text_index(i)
            if len(textss) > 1:
                opt1 = opts[1].for_text_index(i)

            # Accumulate per-sentence output here.
            aline = []

            # If there are multiple texts, determine parts that differ to highlight.
            if len(textss) > 1:
                highlights = list(
                    arrays_different(opt0.input_ids, opt1.input_ids)
                )
            else:
                highlights = [False] * len(opt0.input_ids)

            # Add the visualization of the sentence.
            aline.append(
                self._tokens_stability_instance(
                    opt0,
                    show_id=show_id,
                    show_doi=show_doi,
                    show_text=show_text,
                    highlights=highlights,
                    attributor=attributor
                )
            )

            # Add the visualization of its pair of multiple texts were provided.
            if len(textss) > 1:
                aline.append(
                    self._tokens_stability_instance(
                        opt1,
                        show_id=show_id,
                        show_doi=show_doi,
                        show_text=show_text,
                        highlights=highlights,
                        attributor=attributor
                    )
                )

            # Add the accumulated elements to the final output.
            content.append(self.output.line(self.output.concat(*aline)))

        # Concat/render the entire content.
        return self.output.render(self.output.concat(*content))

    def tokens(
        self,
        texts,
        attributor: AttributionMethod = None,
        show_id: bool = False,
        show_doi: bool = False,
        show_text: bool = False
    ):
        """Visualize a token-based input attribution."""

        return self.tokens_stability(
            texts1=texts,
            attributor=attributor,
            show_id=show_id,
            show_doi=show_doi,
            show_text=show_text
        )

    def _closest_token(self, emb):
        distances = self.embedding_distance(emb)
        # diffs = self.embeddings - emb
        # print(diffs.shape)
        # distances = np.linalg.norm(diffs, ord=2, axis=1)
        # print(distances.shape)
        closest = np.argsort(distances)
        # print(closest.shape)
        return closest[0], distances[closest[0]]
