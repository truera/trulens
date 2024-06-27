import asyncio
import random
import time

from trulens_eval.utils.python import OpaqueWrapper


class Dummy():
    """Perform some operations to inject performance-related characteristics
    into the dummy app.
    
    Args:
        delay: How long to wait in the dummy wait operations.

        alloc: How much memory to allocate in the dummy allocate operations.

        seed: Random seed for the dummy random number generator.
    """

    def __init__(
        self, delay: float = 0.0, alloc: int = 1024, seed: int = 0xdeadbeef
    ):
        self.delay = delay
        self.alloc = alloc

        self._dummy_allocated_data = None

        self.seed = seed
        self.random = random.Random(seed)

    def dummy_wait(self):
        """Wait for a while."""

        if self.delay > 0.0:
            time.sleep(self.delay)

    async def dummy_await(self):
        """Wait for a while."""

        if self.delay > 0.0:
            await asyncio.sleep(self.delay)

    def dummy_allocate(self):
        """Allocate some memory."""

        self._dummy_allocated_data = OpaqueWrapper(
            obj=[True] * self.alloc, e=Exception()
        )
        # OpaqueWrapper will prevent instrumentation or serialization of the
        # contents of this fake data.

        return self._dummy_allocated_data
