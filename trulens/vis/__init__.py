# TODO: Unify visualization parameters for vision and nlp.

# A colormap is a method that given a value between -1.0 and 1.0, returns a quad
# of rgba values, each floating between 0.0 and 1.0.
from typing import Callable, Tuple

import numpy as np


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