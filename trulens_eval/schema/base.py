"""Common/shared serializable classes."""

from __future__ import annotations

import datetime
from typing import Optional

import pydantic

from trulens_eval.utils import serial

MAX_DILL_SIZE: int = 1024 * 1024  # 1MB
"""Max size in bytes of pickled objects."""


class Cost(serial.SerialModel, pydantic.BaseModel):
    """Costs associated with some call or set of calls."""

    n_requests: int = 0
    """Number of requests."""

    n_successful_requests: int = 0
    """Number of successful requests."""

    n_classes: int = 0
    """Number of class scores retrieved."""

    n_tokens: int = 0
    """Total tokens processed."""

    n_stream_chunks: int = 0
    """In streaming mode, number of chunks produced."""

    n_prompt_tokens: int = 0
    """Number of prompt tokens supplied."""

    n_completion_tokens: int = 0
    """Number of completion tokens generated."""

    cost: float = 0.0
    """Cost in USD."""

    def __add__(self, other: 'Cost') -> 'Cost':
        kwargs = {}
        for k in self.model_fields.keys():
            kwargs[k] = getattr(self, k) + getattr(other, k)
        return Cost(**kwargs)

    def __radd__(self, other: 'Cost') -> 'Cost':
        # Makes sum work on lists of Cost.

        if other == 0:
            return self

        return self.__add__(other)


class Perf(serial.SerialModel, pydantic.BaseModel):
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
