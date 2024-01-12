# Index of hacks and other forms of tech debt

This is a (likely incomplete) list of hacks present in the trulens_eval library.
They are likely a source of debugging problems so ideally they can be
addressed/removed in time. This document is to serve as a warning in the
meantime and a resource for hard-to-debug issues when they arise.

In notes below, "HACK###" can be used to find places in the code where the hack
lives.

## Stack inspecting

See `instruments.py` docstring for discussion why these are done.

- We inspect the call stack in process of tracking method invocation. It may be
  possible to replace this with `contextvars`. 

## Method overriding

See `instruments.py` docstring for discussion why these are done.

- We override and wrap methods from other libraries to track their invocation or
  API use. Overriding for tracking invocation is done in the base
  `instruments.py:Instrument` class while for tracking costs are in the base
  `Endpoint` class.

## thread overriding

See `instruments.py` docstring for discussion why these are done.

- "HACK002" -- We override `ThreadPoolExecutor` in `concurrent.futures`.
  
- "HACK007" -- We override `Thread` in `threading`.

### llama-index

- ~~"HACK001" -- `trace_method` decorator in llama_index does not preserve
  function signatures; we hack it so that it does.~~ Fixed as of llama_index
  0.9.26 or near there.
  
### langchain

- "HACK003" -- We override the base class of
  `langchain_core.runnables.config.ContextThreadPoolExecutor` so it uses our
  thread starter.

### pydantic

- "HACK006" -- `endpoint` needs to be added as a keyword arg with default value
  in some `__init__` because pydantic overrides signature without default value
  otherwise.

- "HACK005" -- `model_validate` inside `WithClassInfo` is implemented in
  decorated method because pydantic doesn't call it otherwise.

- We dump attributes marked to be excluded by pydantic except our own classes.
  This is because some objects are of interest despite being marked to exclude.
  Example: `RetrievalQA.retriever` in langchain.

### other

- "HACK004" -- Outdated, need investigation whether it can be removed.

- ~~async/sync code duplication -- Many of our methods are almost identical
  duplicates due to supporting both async and synced versions. Having trouble
  with a working approach to de-duplicated the identical code.~~ Fixed. See
  `utils/asynchro.py`.

- "HACK008" -- async generator -- Some special handling is used for tracking
  costs when async generators are involved. See
  `feedback/provider/endpoint/base.py`.