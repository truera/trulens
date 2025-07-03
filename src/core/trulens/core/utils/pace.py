import asyncio
from collections import deque
from datetime import datetime
from datetime import timedelta
import logging
from threading import Lock
import time
from typing import ClassVar, Deque, Optional

import pydantic
from pydantic import BaseModel
from pydantic import Field
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)


class Pace(BaseModel):
    """Keep a given pace.

    Calls to `Pace.mark` may block until the pace of its returns is kept to a
    constraint: the number of returns in the given period of time cannot exceed
    `marks_per_second * seconds_per_period`. This means the average number of
    returns in that period is bounded above exactly by `marks_per_second`.

    !!! Warning:
        The asynchronous and synchronous methods `amark` and `mark` should not be
        used at the same time. That is, use either the synchronous interface or the
        asynchronous one, but not both.
    """

    marks_per_second: float = 1.0
    """The pace in number of mark returns per second."""

    seconds_per_period: float = 60.0
    """Evaluate pace as the average over this period.

    Assumes that prior to construction of this Pace instance, the period did not
    have any marks called. The longer this period is, the bigger burst of marks
    will be allowed initially and after long periods of no marks.
    """

    seconds_per_period_timedelta: timedelta = Field(
        default_factory=lambda: timedelta(seconds=60.0)
    )
    """The above period as a timedelta."""

    mark_expirations: Deque[datetime] = Field(default_factory=deque)
    """Keep track of returns that happened in the last `period` seconds.

    Store the datetime at which they expire (they become older than `period`
    seconds old).
    """

    max_marks: int
    """The maximum number of marks to keep track in the above deque.

    It is set to (seconds_per_period * returns_per_second) so that the
    average returns per second over period is no more than exactly
    returns_per_second.
    """

    last_mark: datetime = Field(default_factory=datetime.now)
    """Time of the last mark return."""

    _lock: Lock = PrivateAttr(default_factory=Lock)
    """Thread Lock to ensure mark method details run only one at a time."""

    _alock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    """Asyncio Lock to ensure amark method details run only one at a time."""

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True
    )

    _warned: bool = False
    """Whether the long delay warning has already been issued.

    This is to not repeatedly give it.
    """

    def __init__(
        self,
        seconds_per_period: float,
        *args,
        marks_per_second: Optional[float] = None,
        rpm: Optional[float] = None,
        **kwargs,
    ):
        if marks_per_second is None:
            assert (
                rpm is not None
            ), "Either `marks_per_second` or `rpm` must be given."
            marks_per_second = rpm / 60.0
        else:
            assert (
                rpm is None
            ), "Only one of `marks_per_second` or `rpm` can be given."

        max_marks = int(seconds_per_period * marks_per_second)
        if max_marks == 0:
            raise ValueError(
                "Period is too short for the give rate. "
                "Increase `seconds_per_period` or `marks_per_second` (or both)."
            )

        self._ensure_event_loop_exists()

        super().__init__(
            *args,
            seconds_per_period=seconds_per_period,
            seconds_per_period_timedelta=timedelta(seconds=seconds_per_period),
            marks_per_second=marks_per_second,
            max_marks=max_marks,
            **kwargs,
        )

    @staticmethod
    def _ensure_event_loop_exists():
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    async def amark(self) -> float:
        """Return in appropriate pace.

        Blocks until return can happen in the appropriate pace. Returns time in
        seconds since last mark returned.
        """

        async with self._alock:
            while len(self.mark_expirations) >= self.max_marks:
                delay = (
                    self.mark_expirations[0] - datetime.now()
                ).total_seconds()

                if delay >= self.seconds_per_period * 0.5:
                    if not self._warned:
                        self._warned = True
                        logger.warning(
                            """
Pace has a long delay of %s seconds. There might have been a burst of
requests which may become a problem for the receiver of whatever is being paced.
Consider reducing the `seconds_per_period` (currently %s [seconds]) over which to
maintain pace to reduce burstiness. " Alternatively reduce `marks_per_second`
(currently %s [1/second]) to reduce the number of marks
per second in that period.
    """,
                            delay,
                            self.seconds_per_period,
                            self.marks_per_second,
                        )

                if delay > 0.0:
                    await asyncio.sleep(delay)

                self.mark_expirations.popleft()

            prior_last_mark = self.last_mark
            now = datetime.now()
            self.last_mark = now

            # Add to marks the point at which the mark can be removed (after
            # `period` seconds).
            self.mark_expirations.append(
                now + self.seconds_per_period_timedelta
            )

            return (now - prior_last_mark).total_seconds()

    def mark(self) -> float:
        """Return in appropriate pace.

        Blocks until return can happen in the appropriate pace. Returns time in
        seconds since last mark returned.
        """

        with self._lock:
            while len(self.mark_expirations) >= self.max_marks:
                delay = (
                    self.mark_expirations[0] - datetime.now()
                ).total_seconds()

                if delay >= self.seconds_per_period * 0.5:
                    if not self._warned:
                        self._warned = True
                        logger.warning(
                            """
Pace has a long delay of %s seconds. There might have been a burst of
requests which may become a problem for the receiver of whatever is being paced.
Consider reducing the `seconds_per_period` (currently %s [seconds]) over which to
maintain pace to reduce burstiness. " Alternatively reduce `marks_per_second`
(currently %s [1/second]) to reduce the number of marks
per second in that period.
""",
                            delay,
                            self.seconds_per_period,
                            self.marks_per_second,
                        )

                if delay > 0.0:
                    time.sleep(delay)

                self.mark_expirations.popleft()

            prior_last_mark = self.last_mark
            now = datetime.now()
            self.last_mark = now

            # Add to marks the point at which the mark can be removed (after
            # `period` seconds).
            self.mark_expirations.append(
                now + self.seconds_per_period_timedelta
            )

            return (now - prior_last_mark).total_seconds()
