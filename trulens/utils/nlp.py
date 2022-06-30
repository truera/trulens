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
        input_ids = input_accessor(model_inputs)
        device = input_ids.device if hasattr(input_ids, "device") else None
        input_ids = B.as_array(B.clone(input_ids))
        ids = (1 - sum(input_ids == v for v in keep_tokens))

        input_ids[ids] = replacement_token
        input_ids = B.as_tensor(input_ids, dtype=input_ids.dtype, device=device)

        return input_ids

    base_ids: BaselineLike  # expected type

    if ids_to_embeddings is None:
        return base_ids

    def base_embeddings(z: TensorLike = None, model_inputs: ModelInputs = None):
        # :baseline_like
        input_ids = base_ids(z, model_inputs)

        return ids_to_embeddings(input_ids, B.as_tensor(model_inputs.args[0], device=input_ids.device))

    base_embeddings: BaselineLike  # expected type

    return base_ids, base_embeddings
