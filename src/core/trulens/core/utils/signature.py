"""Utilities related to guessing inputs and outputs of functions."""

from inspect import BoundArguments
from inspect import Signature
import logging
from typing import Any, Callable, Dict, Sequence

import pydantic
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils

logger = logging.getLogger(__name__)


def _extract_content(value, content_keys=["content"]):
    """
    Extracts the 'content' from various data types commonly used by libraries
    like OpenAI, Canopy, LiteLLM, etc. This method navigates nested data
    structures (pydantic models, dictionaries, lists) to retrieve the
    'content' field. If 'content' is not directly available, it attempts to
    extract from known structures like 'choices' in a ChatResponse. This
    standardizes extracting relevant text or data from complex API responses
    or internal data representations.

    Args:
        value: The input data to extract content from. Can be a pydantic
                model, dictionary, list, or basic data type.

    Returns:
        The extracted content, which may be a single value, a list of values,
        or a nested structure with content extracted from all levels.
    """
    if isinstance(value, pydantic.BaseModel):
        content = getattr(value, "content", None)
        if content is not None:
            return content

        # If 'content' is not found, check for 'choices' attribute which indicates a ChatResponse
        choices = getattr(value, "choices", None)
        if choices is not None:
            # Extract 'content' from the 'message' attribute of each _Choice in 'choices'
            return [_extract_content(choice.message) for choice in choices]

        # Recursively extract content from nested pydantic models
        try:
            return {
                k: _extract_content(v)
                if isinstance(v, (pydantic.BaseModel, dict, list))
                else v
                for k, v in value.model_dump().items()
            }
        except Exception as e:
            logger.warning(
                "Failed to extract content from pydantic model: %s", e
            )
            # Unsure what is best to do here. Lets just return the exception
            # so that it might reach the user if nothing else gets picked up
            # as main input/output.
            return str(e)

    elif isinstance(value, dict):
        # Check for 'content' key in the dictionary
        for key in content_keys:
            content = value.get(key)
            if content is not None:
                return content

        # Recursively extract content from nested dictionaries
        return {
            k: _extract_content(v) if isinstance(v, (dict, list)) else v
            for k, v in value.items()
        }

    elif isinstance(value, list):
        # Handle lists by extracting content from each item
        return [_extract_content(item) for item in value]

    else:
        return value


def main_input(func: Callable, sig: Signature, bindings: BoundArguments) -> str:
    """Determine (guess) the main input string for a main app call.

    Args:
        func: The main function we are targeting in this determination.

        sig: The signature of the above.

        bindings: The arguments to be passed to the function.

    Returns:
        The main input string.
    """

    if bindings is None:
        raise RuntimeError(
            f"Cannot determine main input of unbound call to {func}: {sig}."
        )

    # ignore self
    all_args = list(
        v for k, v in bindings.arguments.items() if k not in ["self", "_self"]
    )  # llama_index is using "_self" in some places

    # If there is only one string arg, it is a pretty good guess that it is
    # the main input.

    # if have only containers of length 1, find the innermost non-container
    focus = all_args

    while not isinstance(focus, serial_utils.JSON_BASES) and len(focus) == 1:
        focus = focus[0]
        focus = _extract_content(focus, content_keys=["content", "input"])

        if not isinstance(focus, Sequence):
            logger.warning("Focus %s is not a sequence.", focus)
            break

    if isinstance(focus, serial_utils.JSON_BASES):
        return str(focus)

    # Otherwise we are not sure.
    logger.warning(
        "Unsure what the main input string is for the call to %s with args %s.",
        python_utils.callable_name(func),
        bindings,
    )

    # After warning, just take the first item in each container until a
    # non-container is reached.
    focus = all_args
    while not isinstance(focus, serial_utils.JSON_BASES) and len(focus) >= 1:
        focus = focus[0]
        focus = _extract_content(focus)

        if not isinstance(focus, Sequence):
            logger.warning("Focus %s is not a sequence.", focus)
            break

    if isinstance(focus, serial_utils.JSON_BASES):
        return str(focus)

    logger.warning(
        "Could not determine main input/output of %s.", str(all_args)
    )

    return "TruLens: Could not determine main input from " + str(all_args)


def main_output(func: Callable, ret: Any) -> str:
    """Determine (guess) the "main output" string for a given main app call.

    This is for functions whose output is not a string.

    Args:
        func: The main function whose main output we are guessing.

        ret: The return value of the function.
    """

    if isinstance(ret, serial_utils.JSON_BASES):
        return str(ret)

    if isinstance(ret, Sequence) and all(isinstance(x, str) for x in ret):
        # Chunked/streamed outputs.
        return "".join(ret)

    # Use _extract_content to get the content out of the return value
    content = _extract_content(ret, content_keys=["content", "output"])

    if isinstance(content, str):
        return content

    if isinstance(content, float):
        return str(content)

    if isinstance(content, Dict):
        return str(next(iter(content.values()), ""))

    error_message = (
        f"Could not determine main output of {func.__name__}"
        f" from {python_utils.class_name(type(content))} value {content}."
    )

    if isinstance(content, Sequence):
        if len(content) > 0:
            return str(content[0])
        else:
            return error_message

    else:
        logger.warning(error_message)
        return (
            str(content)
            if content is not None
            else (f"TruLens: {error_message}")
        )
