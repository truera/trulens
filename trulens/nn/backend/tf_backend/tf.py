"""Tensorflow Backend
"""
# pylint: disable=no-member
# pylint: disable=not-callable

import numpy as np
import tensorflow as tf

from trulens.nn.backend.load_backend import _ALL_BACKEND_API_FUNCTIONS
__all__ = _ALL_BACKEND_API_FUNCTIONS + ['tf1']

floatX = np.float32
Tensor = tf.Tensor
dim_order = 'channels_last'
channel_axis = 1 if dim_order == 'channels_first' else 3
backend = 'tensorflow'

tf1 = tf.__version__.startswith('1.')


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
    return tf.gradients(scalar, wrt)


def as_array(t, dtype=floatX):
    """
    as_array Convert tensor to numpy array

    Parameters
    ----------
    t : backend.Tensor (tf.Constant)
    dtype : string, optional
        numpy datatype to return, by default floatX

    Returns
    -------
    np.array
        Same contents as t
    """
    if isinstance(t, np.ndarray):
        return t

    if tf1:
        with tf.Session().as_default():
            return t.eval().astype(dtype)
    elif not tf.executing_eagerly():
        with tf.compat.v1.Session() as sess:
            return t.eval()
    else:
        return t.numpy()


def as_tensor(x, device=None):
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
    return tf.constant(x, dtype=floatX)


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
    return tuple(t.shape.as_list())


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
    return t.shape


def expand_dims(t, axis=-1):
    return tf.expand_dims(t, axis)


def reshape(t, shape):
    if isinstance(t, np.ndarray):
        return t.reshape(shape)

    return tf.reshape(t, shape)


def mean(t, axis=None, keepdims=False):
    if isinstance(t, np.ndarray):
        return t.mean(axis=axis, keepdims=keepdims)

    return tf.reduce_mean(t, axis, keepdims)


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

    return tf.reduce_sum(t, axis, keepdims)


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

    return tf.reduce_max(t, axis, keepdims)


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
        return t.min(axis=axis, keepdims=keepdims)

    return tf.reduce_min(t, axis, keepdims)


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

    return tf.maximum(x, y)


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
    return tf.minimum(x, y)


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
    return tf.abs(t)


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
    return tf.ones_like(t, dtype=dtype, name=name)


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
    return tf.zeros_like(t, dtype=dtype, name=name)


def random_normal_like(t, mean=0., var=1.):
    return tf.random.normal(t.shape, mean, stddev=np.sqrt(var))


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
    return tf.sign(t)


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
    return tf.stack(t)


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
    return tf.sigmoid(t)


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
    return tf.nn.softmax(t, axis=axis)


def is_tensor(x):
    """
    is_tensor returns if x is a B.Tensor
    
    Parameters
    ----------
    x : backend.Tensor or other
    """
    if isinstance(x, Tensor):
        return True
    return False
