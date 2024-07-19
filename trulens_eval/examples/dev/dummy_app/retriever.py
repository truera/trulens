import random
import sys

from examples.dev.dummy_app.dummy import Dummy

from trulens_eval.tru_custom_app import instrument


class DummyRetriever(Dummy):
    """Dummy retriever.
    
    Fakes the retrieval of a few chunks.
    """

    def __init__(self, *args, num_contexts: int = 2, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_contexts = num_contexts

    @instrument
    def retrieve_chunks(self, data):
        """Fake chunk retrieval."""

        # Fake delay.
        self.dummy_wait()

        # Fake memory usage.
        temporary = self.dummy_allocate()

        return (
            [
                f"Relevant chunk: {data.upper()}",
                f"Relevant chunk: {data[::-1] * 3}",
                f"Relevant chunk: I allocated {sys.getsizeof(temporary)} bytes to pretend I'm doing something."
            ] * 3
        )[:self.num_contexts]

    @instrument
    async def aretrieve_chunks(self, data):
        """Fake chunk retrieval."""

        # Fake delay.
        await self.dummy_await()

        # Fake memory usage.
        temporary = self.dummy_allocate()

        return (
            [
                f"Relevant chunk: {data.upper()}",
                f"Relevant chunk: {data[::-1] * 3}",
                f"Relevant chunk: I allocated {sys.getsizeof(temporary)} bytes to pretend I'm doing something."
            ] * 3
        )[:self.num_contexts]
