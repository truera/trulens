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

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Colormap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np

from scipy.ndimage.filters import gaussian_filter

from trulens.nn import backend as B

from trulens.nn.attribution import InternalInfluence
from trulens.nn.distributions import PointDoi
from trulens.nn.quantities import InternalChannelQoI
from trulens.nn.slices import Cut, InputCut


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
        if B.dim_order == 'channels_first':
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
            cmap: Colormap = None):
        """
        Configures the default parameters for the `__call__` method (these can 
        be overridden by passing in values to `__call__`).

        Parameters:
            combine_channels:
                If `True`, the attributions will be averaged across the channel
                dimension, resulting in a 1-channel attribution map.

            normalization_type:
                Specifies one of the following configurations for normalizing
                the attributions (each instance is normalized separately):

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
            cmap=None):
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
                visualization, with no batch dimension and the instances in the
                batch tiled along the width and height dimensions. If false, the
                returned array will be reshaped to match `attributions`.

            combine_channels:
                If `True`, the attributions will be averaged across the channel
                dimension, resulting in a 1-channel attribution map. If `None`,
                defaults to the value supplied to the constructor.

            normalization_type:
                Specifies one of the following configurations for normalizing
                the attributions (each instance is normalized separately):

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
            attributions, combine_channels, normalization_type, blur, cmap)

        # Combine the channels if specified.
        if combine_channels:
            attributions = attributions.mean(axis=B.channel_axis, keepdims=True)

        # Blur the attributions so the explanation is smoother.
        if blur:
            attributions = self._blur(attributions, blur)

        # Normalize the attributions.
        attributions = self._normalize(attributions, normalization_type)

        tiled_attributions = self.tiler.tile(attributions)

        # Display the figure:
        _fig = plt.figure() if fig is None else fig

        plt.axis('off')
        plt.imshow(tiled_attributions, cmap=cmap)

        if output_file:
            plt.savefig(output_file, bbox_inches=0)

        if imshow:
            plt.show()

        elif fig is None:
            plt.close(_fig)

        return tiled_attributions if return_tiled else attributions

    def _check_args(
            self, attributions, combine_channels, normalization_type, blur,
            cmap):
        """
        Validates the arguments, and sets them to their default values if they
        are not specified.
        """
        if attributions.ndim != 4:
            raise ValueError(
                '`Visualizer` is inteded for 4-D image-format data. Given '
                'input with dimension {}'.format(attributions.ndim))

        if combine_channels is None:
            combine_channels = self.default_combine_channels

        if not (attributions.shape[B.channel_axis] in (1, 3, 4) or
                combine_channels):

            raise ValueError(
                'To visualize, attributions must have either 1, 3, or 4 color '
                'channels, but `Visualizer` got {} channels.\n'
                'If you are visualizing an internal layer, consider setting '
                '`combine_channels` to True'.format(
                    attributions.shape[B.channel_axis]))

        if normalization_type is None:
            normalization_type = self.default_normalization_type

            if normalization_type is None:
                if combine_channels or attributions.shape[B.channel_axis] == 1:
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
                    ]))

        if blur is None:
            blur = self.default_blur

        if cmap is None:
            cmap = self.default_cmap

        return combine_channels, normalization_type, blur, cmap

    def _normalize(self, attributions, normalization_type, eps=1e-20):
        if normalization_type == 'unnormalized':
            return attributions

        split_by_channel = normalization_type.endswith('sum')

        channel_split = [attributions] if split_by_channel else np.split(
            attributions,
            attributions.shape[B.channel_axis],
            axis=B.channel_axis)

        normalized_attributions = []
        for c_map in channel_split:
            if normalization_type == 'magnitude_max':
                c_map = np.abs(c_map) / (
                    np.abs(c_map).max(axis=(1, 2, 3), keepdims=True) + eps)

            elif normalization_type == 'magnitude_sum':
                c_map = np.abs(c_map) / (
                    np.abs(c_map).sum(axis=(1, 2, 3), keepdims=True) + eps)

            elif normalization_type.startswith('signed_max'):
                postive_max = c_map.max(axis=(1, 2, 3), keepdims=True)
                negative_max = (-c_map).max(axis=(1, 2, 3), keepdims=True)

                # Normalize the postive socres to [0, 1] and negative socresn to
                # [-1, 0].
                normalization_factor = np.where(
                    c_map >= 0, postive_max, negative_max)
                c_map = c_map / (normalization_factor + eps)

                # If positive-centered, normalize so that all scores are in the
                # range [0, 1], with negative scores less than 0.5 and positive
                # scores greater than 0.5.
                if normalization_type.endswith('positive_centered'):
                    c_map = c_map / 2. + 0.5

            elif normalization_type == 'signed_sum':
                postive_max = np.maximum(c_map, 0).sum(
                    axis=(1, 2, 3), keepdims=True)
                negative_max = np.maximum(-c_map, 0).sum(
                    axis=(1, 2, 3), keepdims=True)

                # Normalize the postive socres to ensure they sum to 1 and the
                # negative scores to ensure they sum to -1.
                normalization_factor = np.where(
                    c_map >= 0, postive_max, negative_max)
                c_map = c_map / (normalization_factor + eps)

            elif normalization_type.startswith('unsigned_max'):
                c_map = c_map / (
                    np.abs(c_map).max(axis=(1, 2, 3), keepdims=True) + eps)

                # If positive-centered, normalize so that all scores are in the
                # range [0, 1], with negative scores less than 0.5 and positive
                # scores greater than 0.5.
                if normalization_type.endswith('positive_centered'):
                    c_map = c_map / 2. + 0.5

            elif normalization_type == '01':
                c_map = c_map - c_map.min(axis=(1, 2, 3), keepdims=True)
                c_map = c_map / (c_map.max(axis=(1, 2, 3), keepdims=True) + eps)

            normalized_attributions.append(c_map)

        return np.concatenate(normalized_attributions, axis=B.channel_axis)

    def _blur(self, attributions, blur):
        for i in range(attributions.shape[0]):
            attributions[i] = gaussian_filter(attributions[i], blur)

        return attributions

    def _get_hotcold(self):
        hot = cm.get_cmap('hot', 128)
        cool = cm.get_cmap('cool', 128)
        binary = cm.get_cmap('binary', 128)
        hotcold = np.vstack(
            (
                binary(np.linspace(0, 1, 128)) * cool(np.linspace(0, 1, 128)),
                hot(np.linspace(0, 1, 128))))

        return ListedColormap(hotcold, name='hotcold')


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
            cmap='jet'):
        """
        Configures the default parameters for the `__call__` method (these can 
        be overridden by passing in values to `__call__`).

        Parameters:
            overlay_opacity: float
                Value in the range [0, 1] specifying the opacity for the heatmap
                overlay.

            normalization_type:
                Specifies one of the following configurations for normalizing
                the attributions (each instance is normalized separately):

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
            cmap=cmap)

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
            cmap=None):
        """
        Visualizes the given attributions by overlaying an attribution heatmap 
        over the given image.

        Parameters:
            attributions:
                A `np.ndarray` containing the attributions to be visualized.

            x:
                A `np.ndarray` of instances in the same shape as `attributions`
                corresponding to the instances explained by the given 
                attributions. The visualization will be superimposed onto the
                corresponding set of instances.

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
                visualization, with no batch dimension and the instances in the
                batch tiled along the width and height dimensions. If false, the
                returned array will be reshaped to match `attributions`.

            overlay_opacity: float
                Value in the range [0, 1] specifying the opacity for the heatmap
                overlay. If `None`, defaults to the value supplied to the 
                constructor.

            combine_channels:
                If `True`, the attributions will be averaged across the channel
                dimension, resulting in a 1-channel attribution map. If `None`,
                defaults to the value supplied to the constructor.

            normalization_type:
                Specifies one of the following configurations for normalizing
                the attributions (each instance is normalized separately):

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
            attributions, None, normalization_type, blur, cmap)

        # Combine the channels.
        attributions = attributions.mean(axis=B.channel_axis, keepdims=True)

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
            overlay_opacity = self.overlay_opacity

        # Display the figure:
        _fig = plt.figure() if fig is None else fig

        plt.axis('off')
        plt.imshow(tiled_x)
        plt.imshow(tiled_attributions, alpha=overlay_opacity, cmap=cmap)

        if output_file:
            plt.savefig(output_file, bbox_inches=0)

        if imshow:
            plt.show()

        elif fig is None:
            plt.close(_fig)

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
            positive_only=True):
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
            positive_only=None):

        if attributions.shape != x.shape:
            raise ValueError(
                'Shape of `attributions` {} must match shape of `x` {}'.format(
                    attributions.shape, x.shape))

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
                'input with dimension {}'.format(len(attributions.shape)))

        if combine_channels is None:
            combine_channels = self.default_combine_channels

        if combine_channels:
            attributions = attributions.mean(axis=B.channel_axis, keepdims=True)

        if x.shape[B.channel_axis] not in (1, 3, 4):
            raise ValueError(
                'To visualize, attributions must have either 1, 3, or 4 color '
                'channels, but Visualizer got {} channels.\n'
                'If you are visualizing an internal layer, consider setting '
                '`combine_channels` to True'.format(
                    attributions.shape[B.channel_axis]))

        # Blur the attributions so the explanation is smoother.
        if blur is not None:
            attributions = [gaussian_filter(a, blur) for a in attributions]

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
                ])

        else:
            masks = np.array(attributions)

        # Use the mask on the original image to visualize the explanation.
        attributions = masks * x
        tiled_attributions = self.tiler.tile(attributions)

        if imshow:
            plt.axis('off')
            plt.imshow(tiled_attributions)

            if output_file:
                plt.savefig(output_file, bbox_inches=0)

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
            channel_axis=B.channel_axis,
            agg_fn=None,
            doi=None,
            blur=None,
            threshold=0.5,
            masked_opacity=0.2,
            combine_channels=True,
            use_attr_as_opacity=None,
            positive_only=None):
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

        self.mask_visualizer = MaskVisualizer(
            blur, threshold, masked_opacity, combine_channels,
            use_attr_as_opacity, positive_only)

        self.infl_input = InternalInfluence(
            model, (InputCut(), Cut(layer)),
            InternalChannelQoI(channel, channel_axis, agg_fn),
            PointDoi() if doi is None else doi)

    def __call__(
            self,
            x,
            x_preprocessed=None,
            output_file=None,
            blur=None,
            threshold=None,
            masked_opacity=None,
            combine_channels=None):
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
            x if x_preprocessed is None else x_preprocessed)

        return self.mask_visualizer(
            attrs_input, x, output_file, blur, threshold, masked_opacity,
            combine_channels)
