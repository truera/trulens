"""
Utilities related to core python functionalities.
"""

from inspect import stack


def caller_frame(offset=0):
    return stack()[offset + 1].frame