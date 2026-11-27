import threading

from trulens.core.utils import containers as container_utils


def test_blocking_set_add_after_pop_from_full_set():
    """Popping from a full set frees a slot for a waiting `add`."""
    blocking_set = container_utils.BlockingSet(max_size=1)
    blocking_set.add("first")
    assert blocking_set.pop() == "first"

    added = threading.Event()

    def add_second():
        blocking_set.add("second")
        added.set()

    thread = threading.Thread(target=add_second, daemon=True)
    thread.start()

    assert added.wait(timeout=5), "add() blocked after pop() freed a slot"
    assert blocking_set.pop(blocking=False) == "second"
