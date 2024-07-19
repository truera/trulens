"""Dummy App and Dummy components

The files in this folder code a custom app that resembles an LLM app with
seperate classes for common app components. None of these components make
network calls but pretend to otherwise operate like real components.

!!! Warning
    This example must be run from the git repository as it imports files that
    are not included in a PIP distribution.

# DummyApp options

The custom dummy app has configuration options or call variants for simulating
various behaviours:

- Number of agents and number of tools is configurable via `DummyApp` class
  params/attributes `num_agents` and `num_tools`.

- Simulated delay and memory usage controlled by `Dummy` class `delay` and
  `alloc` attributes. All other components subtype `Dummy` hence also implement
  simulated delays and allocations.

- Non-determinism control via the `Dummy` class `seed` attribute. Randomness in
  some components can be somewhat controlled by this attribute. The control is
  presently incomplete as threading and async usage (described below) is not
  controlled by `seed`.

- Async methods. These begin with the letter "a" followed by the sync method
  name.

- Streaming methods. The main method to app and the generation methods to the
  llm component each have streaming versions (`stream_respond_to_query` and
  `stream`). These produce iterables over strings instead of single strings. The
  async variants add a prefix "a" to these and produce asynchronous iterables.

- Use of parallelism for processing retrieved chunks. This uses either threads
  or async depending on which method variant is called. This is controlled by
  the `DummyApp` class `use_parallel` flag.

- Nested use of custom app inside a custom app component. This is controlled by
  the `DummyAgent` class `use_app` flag.

- Use of trulens_eval to record the invocation of the above nested model. This
  is controlled by the `DummyAgent` class `use_recorder` flag.
"""

from .agent import DummyAgent
from .app import DummyApp
from .dummy import Dummy
from .llm import DummyLLM
from .reranker import DummyReranker
from .retriever import DummyRetriever
from .tool import DummyTool

__all__ = [
    "DummyAgent", "DummyApp", "Dummy", "DummyLLM", "DummyReranker",
    "DummyRetriever", "DummyTool"
]
