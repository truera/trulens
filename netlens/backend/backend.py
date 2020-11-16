import numpy as np


class Backend(AbstractBaseClass):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @property
    def image_dim_order(self):
        return self._image_dim_order

    @property
    def image_channel_axis(self):
        return 3 if self._image_dim_order == 'channels_last' else 1

    def sum(self, t, axis=None, keepdims=False):
        if isinstance(t, np.ndarray):
            return t.sum(axis, keepdims)

        else:
            return self._sum(t, axis, keepdims)

    @abstractmethod
    def _sum(self, t, axis=None, keepdims=False):
        pass
