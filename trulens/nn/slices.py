"""
The *slice*, or layer, of the network provides flexibility over the level of 
abstraction for the explanation. In a low layer, an explanation may highlight 
the edges that were most important in identifying an object like a face, while 
in a higher layer, the explanation might highlight high-level features such as a
nose or mouth. By raising the level of abstraction, explanations that generalize
over larger sets of samples are possible.

Formally, A network, $f$, can be broken into a slice, $f = g \\circ h$, where 
$h$ can be thought of as a pre-processor that computes features, and $g$ can be
thought of as a sub-model that uses the features computed by $h$.
"""
#from __future__ import annotations # Avoid expanding type aliases in mkdocs.

from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Union
from warnings import warn

from trulens.nn.backend import get_backend

# Define some type aliases.
LayerIdentifier = Union[int, str, List[Union[int, str]]]
TensorLike = Union[Any, List[Union[Any]]]


class Cut(object):
    """
    A cut is the primary building block for a slice. It determines an internal
    component of a network to expose. A slice if formed by two cuts.
    """

    def __init__(
            self,
            name: LayerIdentifier,
            anchor: str = 'out',
            accessor: Optional[Callable] = None):
        """
        Parameters:
            name:
                The name or index of a layer in the model, or a list containing
                the names/indices of mutliple layers.

            anchor: 
                Determines whether input (`'in'`) or the output (`'out'`) tensor
                of the spcified layer should be used.

            accessor:
                An accessor function that operates on the layer, mapping the 
                tensor (or list thereof) corresponding to the layer's 
                input/output to another tensor (or list thereof). This can be 
                used to, e.g., extract a particular output from a layer that 
                produces a sequence of outputs. If `accessor` is `None`, the 
                following accessor function will be used: 
                ```python
                lambda t: t[-1] if isinstance(t, list) else t
                ```
        """
        if get_backend().backend == 'pytorch':
            if (isinstance(name, int) or
                (isinstance(name, list) and isinstance(name[0], int))):

                warn(
                    '\n\nPytorch does not have native support for indexed '
                    'layers. Using layer indices is not recommended.\n')

        self.name = name
        self.accessor = accessor
        self.anchor = anchor

    def access_layer(self, layer: TensorLike) -> TensorLike:
        """
        Applies `self.accessor` to the result of collecting the relevant 
        tensor(s) associated with a layer's output.

        Parameters:
            layer:
                The tensor output (or input, if so specified by the anchor) of 
                the layer(s) specified by this cut.

        Returns:
            The result of applying `self.accessor` to the given layer.
        """
        if self.accessor is None:
            return layer[-1] if isinstance(layer, list) else layer

        else:
            layer = (
                layer[0]
                if isinstance(layer, list) and len(layer) == 1 else layer)
            return self.accessor(layer)


class InputCut(Cut):
    """
    Special cut that selects the input(s) of a model.
    """

    def __init__(self, anchor: str = 'in', accessor: Optional[Callable] = None):
        """
        Parameters:
            anchor: 
                Determines whether input (`'in'`) or the output (`'out'`) tensor
                of the spcified layer should be used.

            accessor:
                An accessor function that operates on the layer, mapping the 
                tensor (or list thereof) corresponding to the layer's 
                input/output to another tensor (or list thereof). This can be 
                used to, e.g., extract a particular output from a layer that 
                produces a sequence of outputs. If `accessor` is `None`, the 
                following accessor function will be used: 
                ```python
                lambda t: t[-1] if isinstance(t, list) else t
                ```
        """
        super().__init__(None, anchor, accessor)


class OutputCut(Cut):
    """
    Special cut that selects the output(s) of a model.
    """

    def __init__(
            self, anchor: str = 'out', accessor: Optional[Callable] = None):
        """
        Parameters:
            anchor: 
                Determines whether input (`'in'`) or the output (`'out'`) tensor
                of the spcified layer should be used.

            accessor:
                An accessor function that operates on the layer, mapping the 
                tensor (or list thereof) corresponding to the layer's 
                input/output to another tensor (or list thereof). This can be 
                used to, e.g., extract a particular output from a layer that 
                produces a sequence of outputs. If `accessor` is `None`, the 
                following accessor function will be used: 
                ```python
                lambda t: t[-1] if isinstance(t, list) else t
                ```
        """
        super(OutputCut, self).__init__(None, anchor, accessor)


class LogitCut(Cut):
    """
    Special cut that selects the logit layer of a model. The logit layer must be
    named `'logits'` or otherwise specified by the user to the model wrapper.
    """

    def __init__(
            self, anchor: str = 'out', accessor: Optional[Callable] = None):
        """
        Parameters:
            anchor: 
                Determines whether input (`'in'`) or the output (`'out'`) tensor
                of the spcified layer should be used.

            accessor:
                An accessor function that operates on the layer, mapping the 
                tensor (or list thereof) corresponding to the layer's 
                input/output to another tensor (or list thereof). This can be 
                used to, e.g., extract a particular output from a layer that 
                produces a sequence of outputs. If `accessor` is `None`, the 
                following accessor function will be used: 
                ```python
                lambda t: t[-1] if isinstance(t, list) else t
                ```
        """
        super(LogitCut, self).__init__(None, anchor, accessor)


class Slice(object):
    """
    Class representing a slice of a network. A network, $f$, can be broken
    into a slice, $f = g \\circ h$, where $h$ can be thought of as a 
    pre-processor that computes features, and $g$ can be thought of as a 
    sub-model that uses the features computed by $h$.

    A `Slice` object represents a slice as two `Cut`s, `from_cut` and `to_cut`,
    which are the layers corresponding to the output of $h$ and $g$, 
    respectively.
    """

    def __init__(self, from_cut: Cut, to_cut: Cut):
        """
        Parameters:
            from_cut:
                Cut representing the output of the preprocessing function, $h$,
                in slice, $f = g \\circ h$.

            to_cut:
                Cut representing the output of the sub-model, $g$, in slice, 
                $f = g \\circ h$.
        """
        self._from_cut = from_cut
        self._to_cut = to_cut

    @property
    def from_cut(self) -> Cut:
        """
        Cut representing the output of the preprocessing function, $h$, in 
        slice, $f = g \\circ h$.
        """
        return self._from_cut

    @property
    def to_cut(self) -> Cut:
        """
        Cut representing the output of the sub-model, $g$, in slice, 
        $f = g \\circ h$.
        """
        return self._to_cut

    @staticmethod
    def full_network():
        """
        Returns
        -------
        Slice
            A slice representing the entire model, i.e., :math:`f = g \\circ h`,
            where :math:`h` is the identity function and :math:`g = f`.
        """
        return Slice(InputCut(), OutputCut())
