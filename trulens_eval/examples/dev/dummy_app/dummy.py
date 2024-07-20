import asyncio
import random
import time
from typing import Optional

from trulens_eval.utils.python import OpaqueWrapper


class Dummy():
    """Dummy component base class.
     
    Provides operations to inject performance-related characteristics into the
    dummy app and maintains random number generators for controlling
    non-determinism.

    Args:
        delay: How long to wait in the dummy wait operations.
        
        alloc: How much memory to allocate in the dummy allocate operations.
        
        seed: Random seed for the dummy random number generator.
    """

    def __init__(
        self,
        delay: float = 0.0,
        alloc: int = 0,
        seed: int = 0xdeadbeef
    ):
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
