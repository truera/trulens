# 💣 Tech Debt

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

- "HACK012" -- In the optional imports scheme, we have to make sure that imports
  that happen from outside of trulens raise exceptions instead of
  producing dummies without raising exceptions.

## Method overriding

See `instruments.py` docstring for discussion why these are done.

- We override and wrap methods from other libraries to track their invocation or
  API use. Overriding for tracking invocation is done in the base
  `instruments.py:Instrument` class while for tracking costs are in the base
  `Endpoint` class.

- "HACK009" -- Cannot reliably determine whether a function referred to by an
  object that implements `__call__` has been instrumented. Hacks to avoid
  warnings about lack of instrumentation.

## Thread overriding

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
  decorated method because pydantic doesn't call it otherwise. It is uncertain
  whether this is a pydantic bug.

- We dump attributes marked to be excluded by pydantic except our own classes.
  This is because some objects are of interest despite being marked to exclude.
  Example: `RetrievalQA.retriever` in langchain.

### Other

- "HACK004" -- Outdated, need investigation whether it can be removed.

- ~~async/sync code duplication -- Many of our methods are almost identical
  duplicates due to supporting both async and synced versions. Having trouble
  with a working approach to de-duplicated the identical code.~~ Fixed. See
  `utils/asynchro.py`.

- ~~"HACK008" -- async generator -- Some special handling is used for tracking
  costs when async generators are involved. See
  `feedback/provider/endpoint/base.py`.~~ Fixed in endpoint code.

- "HACK010" -- cannot tell whether something is a coroutine and need additional
  checks in `sync`/`desync`.

- "HACK011" -- older pythons don't allow use of `Future` as a type constructor
  in annotations. We define a dummy type `Future` in older versions of python to
  circumvent this but have to selectively import it to make sure type checking
  and mkdocs is done right.

- "HACK012" -- same but with `Queue`.

- Similarly, we define `NoneType` for older python versions.

- "HACK013" -- when using `from __future__ import annotations` for more
  convenient type annotation specification, one may have to call pydantic's
  `BaseModel.model_rebuild` after all types references in annotations in that file
  have been defined for each model class that uses type annotations that
  reference types defined after its own definition (i.e. "forward refs").

- "HACK014" -- cannot `from trulens_eval import schema` in some places due to
  strange interaction with pydantic. Results in:

  ```python
  AttributeError: module 'pydantic' has no attribute 'v1'
  ```

  It might be some interaction with "from __future__ import annotations" and/or `OptionalImports`.