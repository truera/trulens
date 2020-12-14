"""Pytorch backend
"""
# pylint: disable=no-member
# pylint: disable=not-callable

import numpy as np
import torch

from trulens.nn.backend.load_backend import _ALL_BACKEND_API_FUNCTIONS
__all__ = _ALL_BACKEND_API_FUNCTIONS

floatX = np.float32
Tensor = torch.Tensor
dim_order = 'channels_first'
channel_axis = 1
backend = 'pytorch'


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
    grads = torch.autograd.grad(
        scalar, wrt, retain_graph=True, allow_unused=True, create_graph=True)
    return list(grads)


def as_array(t, dtype=floatX):
    """
    as_array Convert tensor to numpy array

    Parameters
    ----------
    t : backend.Tensor
    dtype : string, optional
        numpy datatype to return, by default floatX

    Returns
    -------
    np.array
        Same contents as t
    """
    if isinstance(t, np.ndarray):
        return t
    return t.cpu().detach().numpy().astype(dtype)


def as_tensor(x, dtype=torch.float32, device=None):
    """
    as_tensor Convert numpy array to tensor

    Parameters
    ----------
    x : np.array
    device : string, optional
        Which device to  associate with the tensor. If None, then
        use the first available cuda device ('cuda:0'), otherwise cpu. 
        By default None

    Returns
    -------
    backend.Tensor
        Same contents as x
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.tensor(x, dtype=dtype).to(device)


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
    return tuple(t.shape)


def shape(t):
    """
    shape Return shape tuple of tensor

    [extended_summary]

    Parameters
    ----------
    t : backend.Tensor

    Returns
    -------
    tuple of int
        Tuple contains the size of each dimension of the tensor.
    """
    return tuple(t.shape)


def expand_dims(t, axis=-1):
    return torch.unsqueeze(t, axis)


def reshape(t, shape):
    if isinstance(t, np.ndarray):
        return t.reshape(shape)

    return torch.reshape(t, shape)


def mean(t, axis=None, keepdims=False):
    if isinstance(t, np.ndarray):
        return t.mean(axis=axis, keepdims=keepdims)

    if axis is not None:
        return torch.mean(t, dim=axis, keepdim=keepdims)
    else:
        return torch.mean(t)


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

    if axis is not None:
        return torch.sum(t, dim=axis, keepdim=keepdims)
    else:
        return torch.sum(t)


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
        by 1. If `keepdims` is `True`, the reduced dimension is retained 
        with length 1., by default False

    Returns
    -------
    backend.Tensor, or tuple
        Max of t, or a tuple of the max of t with the indices

    """
    if isinstance(t, np.ndarray):
        return t.max(axis=axis, keepdims=keepdims)

    return torch.max(t, dim=axis, keepdim=keepdims)[0]


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
        by 1. If `keepdims` is `True`, the reduced dimension is retained 
        with length 1., by default False
    return_indices : bool, optional
        if `return_indices` is `True`, returns a tuple (values, indices) 
        where values is the minimum value of each row of the input tensor 
        in the given dimension dim. And indices is the index location of 
        each minimum value found (argmin). by default False

    Returns
    -------
    backend.Tensor, or tuple
        Min of t, or a tuple of the min of t with the indices

    """
    if isinstance(t, np.ndarray):
        return t.min(axis=axis, keepdims=keepdims)

    return torch.min(t, dim=axis, keepdim=keepdims)[0]


def maximum(x, y):
    """
    maximum Element-wise maximum of two input tensors
    
    Parameters
    ----------
    x : backend.Tensor
    y : backend.Tensor
    
    Returns
    -------
    backend.Tensor
        Element-wise maximum tensor
    """
    return torch.max(x, y)


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
    return torch.min(x, y)


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
    return torch.abs(t)


def ones_like(t, dtype=None, requires_grad=False):
    """
    ones_like Create a tensor of ones with the same shape of the input tensor
    on the same device
    
    Parameters
    ----------
    t : backend.Tensor
    dtype : torch.dtype, optional
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
    return torch.ones_like(t, dtype=dtype, requires_grad=requires_grad)


def zeros_like(t, dtype=None, requires_grad=False):
    """
    zeros_like Create a tensor of ones with the same shape of the input tensor
    on the same device
    
    Parameters
    ----------
    t : backend.Tensor
    dtype : torch.dtype, optional
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
    if not is_tensor(t):
        t = as_tensor(t)
    return torch.zeros_like(t, dtype=dtype, requires_grad=requires_grad)


def random_normal_like(t, mean=0., var=1.):
    return torch.empty_like(t).normal_(mean, std=np.sqrt(var))


def clone(t):
    """
    clone Return a tensor with the same content as the input tensor.

    Parameters
    ----------
    t : backend.Tensor

    Returns
    -------
    backend.Tensor
        A tensor with the same content as the input tensor.

    """
    return t.clone()


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
    return torch.sign(t)


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
    return torch.sigmoid(t)


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
    return torch.stack(t)


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
    return torch.nn.Softmax(dim=axis)(t)


def is_tensor(x):
    """
    is_tensor returns if x is a B.Tensor
    
    Parameters
    ----------
    x : backend.Tensor or other
    """
    return isinstance(x, Tensor)
