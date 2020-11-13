# pylint:disable=unused-wildcard-import
#from __future__ import absolute_import
from netlens import _NETLENS_BACKEND
import importlib

import os

if _NETLENS_BACKEND is not None:
    _BACKEND = _NETLENS_BACKEND
elif 'NETLENS_BACKEND' in os.environ.keys():
    _BACKEND = os.environ['NETLENS_BACKEND']
else:
    _BACKEND = 'pytorch'

_ALL_BACKEND_API_FUNCTIONS = [
    'dim_order',
    'channel_axis',
    'Tensor',
    'floatX',
    'backend',
    'gradient',
    'as_array',
    'as_tensor',
    'is_tensor',
    'int_shape',
    'shape',
    'expand_dims',
    'reshape',
    'mean',
    'sum',
    'abs',
    'max',
    'ones_like',
    'zeros_like',
    'random_normal_like',
    'clone',
    'stack',
    'sign',
    'sigmoid',
    'softmax',
    'maximum',
    'minimum',
]
