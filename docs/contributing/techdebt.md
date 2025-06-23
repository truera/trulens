# 💣 Tech Debt

This is a (likely incomplete) list of hacks present in the TruLens library.
They are likely a source of debugging problems, so ideally they can be
addressed/removed in time. This document is to serve as a warning in the
meantime and a resource for hard-to-debug issues when they arise.

In the notes below, "HACK###" can be used to find places in the code where the hack
lives.

## Stack inspecting

See `instruments.py` docstring for a discussion of why these are done.

- __Stack walking removed in favor of contextvars in 1.0.3__. We inspect the
  call stack in the process of tracking method invocation. It may be possible to
  replace this with `contextvars`.

- "HACK012" -- In the optional imports scheme, we have to ensure that imports
  from outside of TruLens raise exceptions instead of
  producing dummy objects silently.

## Method overriding

See `instruments.py` docstring for discussion why these are done.

- We override and wrap methods from other libraries to track their invocation or
  API use. Overriding for tracking invocation is done in the base
  `instruments.py:Instrument` class, while overriding for tracking costs is done in the base
  `Endpoint` class.

- "HACK009" -- Cannot reliably determine whether a function referred to by an
  object that implements `__call__` has been instrumented. Hacks to avoid
  warnings about lack of instrumentation.

## Thread overriding

See `instruments.py` docstring for discussion why these are done.

- "HACK002" -- We override `ThreadPoolExecutor` in `concurrent.futures`.

- "HACK007" -- We override `Thread` in `threading`.

### LlamaIndex

- __Fixed as of llama_index 0.9.26 or near there.__ "HACK001" -- `trace_method`
  decorator in llama_index does not preserve function signatures; we hack it so
  that it does.

### LangChain

- "HACK003" -- We override the base class of
  `langchain_core.runnables.config.ContextThreadPoolExecutor` so that it uses our
  thread starter.

### Pydantic

- "HACK006" -- `endpoint` needs to be added as a keyword arg with default value
  in some `__init__` methods because Pydantic would otherwise override the signature without a default value.

- "HACK005" -- `model_validate` inside `WithClassInfo` is implemented in
  decorated method because Pydantic doesn't call it otherwise. It is uncertain
  whether this is a Pydantic bug.

- We dump attributes marked to be excluded by Pydantic except our own classes.
  This is because some objects are of interest despite being marked to exclude.
  Example: `RetrievalQA.retriever` in LangChain.

### Other

- "HACK004" -- Outdated, need investigation whether it can be removed.

- __Partially fixed with asynchro module:__ async/sync code duplication -- Many
  of our methods are almost identical duplicates due to supporting both async
  and sync versions. Having trouble with a working approach to de-duplicated
  the identical code.

- __Fixed in endpoint code:__ "HACK008" -- async generator -- We implement special
  handling to track costs when async generators are involved. See
  `feedback/provider/endpoint/base.py`.

- "HACK010" -- We cannot tell whether something is a coroutine and therefore need additional
  checks in `sync`/`desync`.

- "HACK011" -- older versions of Python don't allow the use of `Future` as a type constructor
  in annotations. We define a dummy type `Future` in older versions of Python to
  circumvent this but have to selectively import it to make sure type checking
  and mkdocs is done right.

- "HACK012" -- same but with `Queue`.

- Similarly, we define `NoneType` for older Python versions that don't include it natively.

- "HACK013" -- when using `from __future__ import annotations` for more
  convenient type annotation specification, one may have to call Pydantic's
  `BaseModel.model_rebuild` after all types references in annotations in that file
  have been defined for each model class that uses type annotations that
  reference types defined after its own definition (i.e. "forward refs").

- "HACK014" -- cannot `from trulens import schema` in some places due to
  strange interaction with Pydantic. Results in:

    ```python
    AttributeError: module 'pydantic' has no attribute 'v1'
    ```

    It might be some interaction with `from __future__ import annotations` and/or
    `OptionalImports`.
