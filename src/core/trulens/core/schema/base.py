"""Common/shared serializable classes."""

from __future__ import annotations

import datetime
from typing import Optional

import pydantic
from trulens.core.utils import serial as serial_utils

MAX_DILL_SIZE: int = 1024 * 1024  # 1MB
"""Max size in bytes of pickled objects."""


class Cost(serial_utils.SerialModel, pydantic.BaseModel):
    """Costs associated with some call or set of calls."""

    n_requests: int = 0
    """Number of requests.

    To increment immediately when a request is made.
    """

    n_successful_requests: int = 0
    """Number of successful requests.

    To increment only after a response is processed and it indicates success.
    """

    n_completion_requests: int = 0
    """Number of completion requests.

    To increment immediately when a completion request is made.
    """

    n_classification_requests: int = 0
    """Number of classification requests.

    To increment immediately when a classification request is made.
    """

    n_classes: int = 0
    """Number of class scores retrieved.

    To increment for each class in a successful classification response.
    """

    n_embedding_requests: int = 0
    """Number of embedding requests.

    To increment immediately when an embedding request is made.
    """

    n_embeddings: int = 0
    """Number of embeddings retrieved.

    To increment for each embedding vector returned by an embedding request.
    """

    n_tokens: int = 0
    """Total tokens processed.

    To increment by the number of input(prompt) and output tokens in completion
    requests. While the input part of this could be incremented upon a request,
    the actual count is not easy to determine due to tokenizer variations and
    instead is usually seen in the response. Also, we want to count only tokens
    for successful requests that incur a cost to the user.
    """

    n_stream_chunks: int = 0
    """In streaming mode, number of chunks produced.

    To increment for each chunk in a streaming response. This does not need to
    wait for completion of the responses.
    """

    n_prompt_tokens: int = 0
    """Number of prompt tokens supplied.

    To increment by the number of tokens in the prompt of a completion request.
    This is visible in the response though and should only count successful
    requests.
    """

    n_completion_tokens: int = 0
    """Number of completion tokens generated.

    To increment by the number of tokens in the completion of a completion request.
    """

    n_cortex_guardrails_tokens: int = 0
    """Number of guardrails tokens generated. This is only available for
    requests instrumented by the Cortex endpoint."""

    cost: float = 0.0
    """Cost in [cost_currency].

    This may not always be available or accurate.
    """

    cost_currency: str = "USD"

    def __add__(self, other: "Cost") -> "Cost":
        kwargs = {
            k: getattr(self, k) + getattr(other, k)
            if k != "cost_currency"
            and isinstance(getattr(self, k), (int, float))
            else getattr(other, k)
            for k in type(self).model_fields.keys()
        }
        if other.cost_currency != self.cost_currency:
            if self.cost == 0:
                kwargs["cost_currency"] = other.cost_currency
            elif other.cost == 0:
                kwargs["cost_currency"] = self.cost_currency
            else:
                raise ValueError(
                    f"Cannot add costs with different currencies: {self.cost_currency} and {other.cost_currency}!"
                )
        return Cost(**kwargs)

    def __radd__(self, other: "Cost") -> "Cost":
        # Makes sum work on lists of Cost.

        if other == 0:
            return self

        return self.__add__(other)


class Perf(serial_utils.SerialModel, pydantic.BaseModel):
    """Performance information.

    Presently only the start and end times, and thus latency.
    """

    start_time: datetime.datetime
    """Datetime before the recorded call."""

    end_time: datetime.datetime
    """Datetime after the recorded call."""

    @staticmethod
    def min():
        """Zero-length span with start and end times at the minimum datetime."""

        return Perf(
            start_time=datetime.datetime.min, end_time=datetime.datetime.min
        )

    @staticmethod
    def now(latency: Optional[datetime.timedelta] = None) -> Perf:
        """Create a `Perf` instance starting now and ending now plus latency.

        Args:
            latency: Latency in seconds. If given, end time will be now plus
                latency. Otherwise end time will be a minimal interval plus start_time.
        """

        start_time = datetime.datetime.now()
        if latency is not None:
            end_time = start_time + latency
        else:
            end_time = start_time + datetime.timedelta(microseconds=1)

        return Perf(start_time=start_time, end_time=end_time)

    @property
    def latency(self):
        """Latency in seconds."""
        return self.end_time - self.start_time


# HACK013: Need these if using __future__.annotations .
Cost.model_rebuild()
Perf.model_rebuild()
