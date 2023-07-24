"""
Utilities related to core python functionalities.
"""

from inspect import stack
from typing import Callable, TypeVar


T = TypeVar("T")
Thunk = Callable[[], T]


def caller_frame(offset=0):
    return stack()[offset + 1].frame