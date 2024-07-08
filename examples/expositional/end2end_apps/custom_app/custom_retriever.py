import sys
import time

from trulens_eval.tru_custom_app import instrument


class CustomRetriever:

    def __init__(self, delay: float = 0.015, alloc: int = 1024 * 1024):
        self.delay = delay
        self.alloc = alloc

    # @instrument
    def retrieve_chunks(self, data):
        temporary = [0x42] * self.alloc

        if self.delay > 0.0:
            time.sleep(self.delay)

        return [
            f"Relevant chunk: {data.upper()}", f"Relevant chunk: {data[::-1]}",
            f"Relevant chunk: I allocated {sys.getsizeof(temporary)} bytes to pretend I'm doing something."
        ]
