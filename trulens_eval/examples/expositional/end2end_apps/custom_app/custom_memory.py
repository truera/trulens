import sys
import time

from trulens.tru_custom_app import instrument


class CustomMemory:

    def __init__(self, delay: float = 0.0, alloc: int = 1024 * 1024):
        self.alloc = alloc
        self.delay = delay

        # keep a chunk of data allocated permentantly:
        self.temporary = [0x42] * self.alloc

        self.messages = []

    def remember(self, data: str):
        if self.delay > 0.0:
            time.sleep(self.delay)

        self.messages.append(
            data +
            f" and I'm keeping around {sys.getsizeof(self.temporary)} bytes"
        )
