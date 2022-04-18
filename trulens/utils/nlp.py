"""Utilities for using trulens with NLP models. """

from typing import Callable, Optional, Set

from trulens.nn.backend import get_backend
from trulens.utils.typing import BaselineLike
from trulens.utils.typing import ModelInputs
from trulens.utils.typing import Tensor
from trulens.utils.typing import TensorLike


def token_baseline(
    keep_tokens: Set[int], replacement_token: int,
    input_accessor: Callable[[ModelInputs], Tensor],
    ids_to_embeddings: Optional[Callable[[int], Tensor]]
):
    """
    Utility for constructing baselines for models with an embedding layer.
    Replace all tokens except those in the keep_tokens set with the specified
    replacement token. Returns methods for producing baseline ids and if
    ids_to_embeddings was given, baseline embeddings.

    Parameters:

        keep_tokens: Set[int]
            The set of tokens by id to preserve in the baseline.

        replacement_token: int
            For tokens not in the keep_tokens set, the token to replace them
            with.

        ids_to_embeddings: Callable[[int], Tensor]

    """

    B = get_backend()

    def base_ids(z: TensorLike = None, model_inputs: ModelInputs = None):
        # :baseline_like
        input_ids = B.clone(input_accessor(model_inputs))

        ids = (1 - sum(input_ids == v for v in keep_tokens)).bool()

        input_ids[ids] = replacement_token

        return input_ids

    base_ids: BaselineLike  # expected type

    if ids_to_embeddings is None:
        return base_ids

    def base_embeddings(z: TensorLike = None, model_inputs: ModelInputs = None):
        # :baseline_like
        input_ids = base_ids(z, model_inputs)

        return ids_to_embeddings(input_ids)

    base_embeddings: BaselineLike  # expected type

    return base_ids, base_embeddings
