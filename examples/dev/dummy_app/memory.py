import sys

from trulens.instrument.custom import instrument

from examples.dev.dummy_app.dummy import Dummy


class DummyMemory(Dummy):
    """Dummy memory implementation that marely apends memories to a list."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.memory_type = "dummy-memory"

        # Fake memory allocation:
        self.temporary = self.dummy_allocate()

        self.messages = []

    @instrument
    def remember(self, data: str):
        """Add a piece of data to memory."""

        # Fake delay.
        self.dummy_wait()

        self.messages.append(
            data
            + f" and I'm keeping around {sys.getsizeof(self.temporary)} bytes"
        )

    @instrument
    async def aremember(self, data: str):
        """Add a piece of data to memory."""

        # Fake delay.
        await self.dummy_await()

        self.messages.append(
            data
            + f" and I'm keeping around {sys.getsizeof(self.temporary)} bytes"
        )
