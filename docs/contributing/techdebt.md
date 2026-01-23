# 💣 Tech Debt

This is a (likely incomplete) list of hacks present in the TruLens library.
They are likely a source of debugging problems, so ideally they can be
addressed/removed in time. This document is to serve as a warning in the
meantime and a resource for hard-to-debug issues when they arise.

In the notes below, "HACK###" can be used to find places in the code where the hack
lives.

## OpenTelemetry Migration

As of TruLens 1.x, migrated instrumentation to OpenTelemetry (OTEL).
The new OTEL-based instrumentation is in `trulens.core.otel.instrument` and
`trulens.experimental.otel_tracing`. This migration addresses several of the
tech debt items below:

- Context propagation is now handled by OTEL's context API instead of custom
  stack inspection
- Span-based tracing provides standardized instrumentation
- OTEL exporters allow flexible data export to various backends

Many of the hacks below relate to the legacy instrumentation system and are
candidates for removal now that OTEL is the primary instrumentation approach.

## Stack Inspecting

See `instruments.py` docstring for a discussion of why these are done.

- __Addressed with contextvars and OTEL context__. Stack walking was removed in
  favor of contextvars in 1.0.3. OTEL now handles context propagation.

- "HACK012" -- In the optional imports scheme, we have to ensure that imports
  from outside of TruLens raise exceptions instead of
  producing dummy objects silently.

## Method Overriding

See `instruments.py` docstring for discussion why these are done.

- We override and wrap methods from other libraries to track their invocation or
  API use. The OTEL-based instrumentation uses the `@instrument` decorator from
  `trulens.core.otel.instrument`. Legacy instrumentation in `instruments.py` is
  still present but being phased out.

- "HACK009" -- Cannot reliably determine whether a function referred to by an
  object that implements `__call__` has been instrumented. Hacks to avoid
  warnings about lack of instrumentation.

## Thread Overriding

See `instruments.py` docstring for discussion why these are done.

- "HACK002" -- We override `ThreadPoolExecutor` in `concurrent.futures`.

- "HACK007" -- We override `Thread` in `threading`.

These are less necessary now that OTEL provides context propagation across
threads, but are still present for backwards compatibility.

### LlamaIndex

- __Fixed as of llama_index 0.9.26.__ "HACK001" -- `trace_method` decorator in
  llama_index previously did not preserve function signatures.

### LangChain

- __Removed.__ "HACK003" -- Previously overrode the base class of
  `langchain_core.runnables.config.ContextThreadPoolExecutor`. This hack is no
  longer in the codebase.

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

- __Disabled.__ "HACK004" -- Related to provider loading in feedback functions.
  The hack is commented out in `pyschema.py` as we now have different providers
  that may need to be selected.

- __Partially fixed with asynchro module:__ async/sync code duplication -- Many
  of our methods are almost identical duplicates due to supporting both async
  and sync versions. Having trouble with a working approach to de-duplicated
  the identical code.

- __Fixed in endpoint code:__ "HACK008" -- async generator -- We implement special
  handling to track costs when async generators are involved. See
  `feedback/provider/endpoint/base.py`.

- "HACK010" -- Cannot tell whether something is a coroutine and need additional
  checks in `sync`/`desync`.

- __May be removable (Python >= 3.9 required).__ "HACK011" -- older versions of
  Python don't allow use of `Future` as a type constructor in annotations. We
  define a dummy type `Future` in older versions of Python to circumvent this.
  Since TruLens now requires Python >= 3.9, this may be removable.

- __May be removable (Python >= 3.9 required).__ "HACK012" -- same but with
  `Queue`. Note: This is different from HACK012 in the optional imports context.

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
