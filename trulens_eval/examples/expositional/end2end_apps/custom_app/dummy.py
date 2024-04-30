import asyncio
import time

class Dummy():
    """Perform some operations to inject performance-related characteristics
    into the dummy app.
    
    Args:
        delay: How long to wait in the dummy wait operations.

        alloc: How much memory to allocate in the dummy allocate operations.
    """
    def __init__(self, delay: float = 0.0, alloc: int = 1024 * 1024):
        self.delay = delay
        self.alloc = alloc

        self.dummy_allocated_data = None

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

        self.dummy_allocated_data = [True] * self.alloc

        return self.dummy_allocated_data
