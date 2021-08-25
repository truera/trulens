"""Keras backend
"""
# pylint: disable=no-member
# pylint: disable=not-callable

import numpy as np
import os

from trulens.nn.backend import _ALL_BACKEND_API_FUNCTIONS, Backend

__all__ = _ALL_BACKEND_API_FUNCTIONS

if 'TRULENS_BACKEND' in os.environ.keys():
    _TRULENS_BACKEND = os.environ['TRULENS_BACKEND']

backend = Backend.from_name(_TRULENS_BACKEND)

if backend == Backend.TF_KERAS:
    import tensorflow.keras.backend as K
    import tensorflow as tf
else:
    import keras.backend as K

floatX = K.floatx()
Tensor = type(K.constant((1, 1), dtype=floatX))
TensorVar = type(K.zeros((1, 1), dtype=floatX))
dim_order = K.image_data_format()
channel_axis = 1 if dim_order == 'channels_first' else 3


def gradient(scalar, wrt):
    """
    gradient Gradient of a function with respect to a tensor.

    Parameters
    ----------
    scalar : backend.Tensor
        The scalar tensor result of a computation for which the gradient will be 
        computed.
    wrt : backend.Tensor
        Tensor that the gradient is taken with respect to.

    Returns
    -------
    list
        A list of computed gradient; same shape as wrt
    """
    if not isinstance(wrt, list):
        wrt = [wrt]

    grads = K.gradients(scalar, wrt)

    if not isinstance(grads, list):
        return [grads]
    else:
        return grads


def as_array(t, dtype=None):
    """
    as_array Convert tensor to numpy array

    Parameters
    ----------
    t : backend.Tensor
    dtype : string, optional
        numpy datatype to return, derived from `t` by default

    Returns
    -------
    np.array
        Same contents as t
    """
    return K.get_value(t) if dtype is None else K.get_value(t).astype(dtype)


def as_tensor(x, dtype=None, device=None):
    """
    as_tensor Convert numpy array to tensor

    Parameters
    ----------
    x : np.array
    device : string, optional
        Ignored

    Returns
    -------
    backend.Tensor
        Same contents as x
    """
    if dtype is None and x.dtype.kind == 'f':
        dtype = floatX

    return K.constant(x, dtype=dtype)


def int_shape(t):
    """
    int_shape Return shape tuple of tensor

    [extended_summary]

    Parameters
    ----------
    t : backend.Tensor

    Returns
    -------
    tuple of int
        Tuple contains the size of each dimension of the tensor.
    """
    return K.int_shape(t)


def shape(t):
    """
    shape Return shape of a tensor as a tensor

    [extended_summary]

    Parameters
    ----------
    t : backend.Tensor

    Returns
    -------
    tuple of int
        Tuple contains the size of each dimension of the tensor.
    """
    return K.shape(t)


def expand_dims(t, axis=-1):
    return K.expand_dims(t, axis)


def reshape(t, shape):
    if isinstance(t, np.ndarray):
        return t.reshape(shape)

    return K.reshape(t, shape)


def mean(t, axis=None, keepdims=False):
    if isinstance(t, np.ndarray):
        return t.mean(axis=axis, keepdims=keepdims)

    return K.mean(t, axis, keepdims)


def sum(t, axis=None, keepdims=False):
    """
    sum Sum of values in tensor

    Parameters
    ----------
    t : backend.Tensor
    axis : int or list of ints, optional
        The dimensions to sum over, or if None sum over all
        dimensions. By default None
    keepdims : bool, optional
        If `keepdims` is `False`, the rank of the tensor is reduced 
        by 1 for each element in axis. If `keepdims` is `True`, 
        the reduced dimension is  retained with length 1., by default False

    Returns
    -------
    backend.Tensor
        Sum of t
    """
    if isinstance(t, np.ndarray):
        return t.sum(axis=axis, keepdims=keepdims)

    return K.sum(t, axis, keepdims)


def max(t, axis=None, keepdims=False):
    """
    max Maximum values of tensor, element-wise

    Parameters
    ----------
    t : backend.Tensor
    axis : int, optional
        The dimension to max over, or if None max over all
        dimensions. By default None
    keepdims : bool, optional
        If `keepdims` is `False`, the rank of the tensor is reduced 
        by 1. If `keepdims` is `True`, the reduced dimension is  retained 
        with length 1., by default False

    Returns
    -------
    backend.Tensor
        Max of t
    """
    if isinstance(t, np.ndarray):
        return t.max(axis=axis, keepdims=keepdims)

    return K.max(t, axis, keepdims)


def min(t, axis=None, keepdims=False):
    """
    min Minimum values of tensor, element-wise 

    Parameters
    ----------
    t : backend.Tensor
    axis : int, optional
        The dimension to min over, or if None min over all
        dimensions. By default None
    keepdims : bool, optional
        If `keepdims` is `False`, the rank of the tensor is reduced 
        by 1. If `keepdims` is `True`, the reduced dimension is  retained 
        with length 1., by default False

    Returns
    -------
    backend.Tensor
        Min of t
    """
    if isinstance(t, np.ndarray):
        return t.sum(axis=axis, keepdims=keepdims)

    return K.min(t, axis, keepdims)


def maximum(x, y):
    """
    maximum Element-wise max of two input tensors
    
    Parameters
    ----------
    x : backend.Tensor
    y : backend.Tensor
    
    Returns
    -------
    backend.Tensor
        Element-wise maximum tensor
    """

    return K.maximum(x, y)


def minimum(x, y):
    """
    minimum Element-wise minimum of two input tensors
    
    Parameters
    ----------
    x : backend.Tensor
    y : backend.Tensor
    
    Returns
    -------
    backend.Tensor
        Element-wise minimum tensor
    """
    return K.minimum(x, y)


def abs(t):
    """
    abs Absolute value of tensor, element-wise

    Parameters
    ----------
    t : backend.Tensor

    Returns
    -------
    backend.Tensor
        Each coordinate contains absolute value of corresponding input
    """
    return K.abs(t)


def ones_like(t, dtype=None, name=None):
    """
    ones_like Create a tensor of ones with the same shape of the input tensor
    on the same device
    
    Parameters
    ----------
    t : backend.Tensor
    dtype : string, optional
        The desired data type of returned Tensor. If None, 
        defaults to the dtype of input, by default None
    requires_grad : bool, optional
        If autograd should record operations on the returned 
        tensor, by default False
    
    Returns
    -------
    backend.Tensor
        A tensor of ones has the same shape of input tensor
    """
    return K.ones_like(t, dtype=dtype, name=name)


def zeros_like(t, dtype=None, name=None):
    """
    zeros_like Create a tensor of ones with the same shape of the input tensor
    on the same device
    
    Parameters
    ----------
    t : backend.Tensor
    dtype : string, optional
        The desired data type of returned Tensor. If None, 
        defaults to the dtype of input, by default None
    requires_grad : bool, optional
        If autograd should record operations on the returned 
        tensor, by default False
    
    Returns
    -------
    backend.Tensor
        A tensor of zeros has the same shape of input tensor
    """
    return K.zeros_like(t, dtype=dtype, name=name)


def random_normal_like(t, mean=0., var=1.):
    return K.random_normal(K.shape(t), mean, stddev=np.sqrt(var))


def clone(t, name=None):
    """
    clone Return a tensor with the same content as the input tensor.

    Parameters
    ----------
    t : backend.Tensor
    name: string, optional
        Name for the variable to create., by default None
    
    Returns
    -------
    backend.Tensor
        A tensor with the same content as the input tensor.
    """
    return identity(t, name=name)


def identity(t, name=None):
    """
    identity An alias function for Keras naming convention 

    Parameters
    ----------
    t : backend.Tensor
    name: string, optional
        Name for the variable to create., by default None
    
    Returns
    -------
    backend.Tensor
        A tensor of zeros has the same shape of input tensor
    """
    if backend == Backend.KERAS:
        return K.identity(t, name=name)
    elif backend == Backend.TF_KERAS:
        return tf.identity(t, name=name)


def sign(t):
    """
    sign Return a tensor with the element-wise sign of the input tensor

    Parameters
    ----------
    t : backend.Tensor

    Returns
    -------
    backend.Tensor

    """
    return K.sign(t)


def stack(t):
    """
    stack Return a tensor with the input tensors vertically stacked

    Parameters
    ----------
    t : list of backend.Tensors

    Returns
    -------
    backend.Tensor

    """
    return K.stack(t)


def sigmoid(t, axis=None):
    """
    sigmoid Sigmoid function 

    Parameters
    ----------
    t : backend.Tensor
    axis : int
        Ignore

    Returns
    -------
    backend.Tensor

    """
    return K.sigmoid(t)


def softmax(t, axis=-1):
    """
    softmax Softmax of a tensor
    
    Parameters
    ----------
    t : backend.Tensor
    axis : int, optional
        The dimension softmax would be performed on. 
        The default is -1 which indicates the last dimension, by default -1
    
    Returns
    -------
    backend.Tensor
    """
    return K.softmax(t, axis=axis)


def is_tensor(x):
    """
    is_tensor returns if x is a get_backend().Tensor
    
    Parameters
    ----------
    x : backend.Tensor or other
    """
    try:
        is_keras_tensor = K.is_keras_tensor(x)
    except:
        is_keras_tensor = False

    return isinstance(x, Tensor) or isinstance(x, TensorVar) or is_keras_tensor
