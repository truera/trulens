import sys
import time

from trulens.core.app.custom import instrument


class CustomLLM:
    def __init__(
        self, model: str = "derp", delay: float = 0.01, alloc: int = 1024 * 1024
    ):
        self.model = model
        self.delay = delay
        self.alloc = alloc

    @instrument
    def generate(self, prompt: str):
        if self.delay > 0.0:
            time.sleep(self.delay)

        temporary = [0x42] * self.alloc

        return (
            "herp "
            + prompt[::-1]
            + f" derp and {sys.getsizeof(temporary)} bytes"
        )
