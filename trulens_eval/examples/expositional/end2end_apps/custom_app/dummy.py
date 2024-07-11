import asyncio
import random
import time
from typing import Optional

from trulens_eval.utils.python import OpaqueWrapper


class Dummy():
    """Perform some operations to inject performance-related characteristics
    into the dummy app.
    """

    DEFAULT_ALLOC: int = 0
    """How much memory to allocate in the dummy allocate operations."""

    DEFAULT_DELAY: float = 0.0
    """How long to wait in the dummy wait operations."""

    DEFAULT_SEED: int = 0xdeadbeef
    """Random seed for the dummy random number generator."""

    def __init__(
        self,
        delay: Optional[float] = None,
        alloc: Optional[int] = None,
        seed: Optional[int] = None
    ):
        if delay is None:
            delay = Dummy.DEFAULT_DELAY
        if alloc is None:
            alloc = Dummy.DEFAULT_ALLOC
        if seed is None:
            seed = Dummy.DEFAULT_SEED

        self.delay = delay
        self.alloc = alloc
        self.seed = seed
        self.random = random.Random(seed)

        self._dummy_allocated_data = None

    def dummy_wait(self, delay: Optional[float] = None):
        """Wait for a while."""

        if delay is None:
            delay = self.delay

        if delay > 0.0:
            time.sleep(delay)

    async def dummy_await(self, delay: Optional[float] = None):
        """Wait for a while."""

        if delay is None:
            delay = self.delay

        if delay > 0.0:
            await asyncio.sleep(delay)

    def dummy_allocate(self):
        """Allocate some memory."""

        self._dummy_allocated_data = OpaqueWrapper(
            obj=[True] * self.alloc, e=Exception()
        )
        # OpaqueWrapper will prevent instrumentation or serialization of the
        # contents of this fake data.

        return self._dummy_allocated_data
