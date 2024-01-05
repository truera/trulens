# Index of hacks and other forms of bitrot

This is a (likely incomplete) list of hacks present in the trulens_eval library.
They are likely a source of debugging problems so ideally they can be
addressed/removed in time. This document is to serve as a warning in the
meantime and a resource for hard-to-debug issues when they arise.

In notes below, "HACK###" can be used to find places in the code where the hack
lives.

## Method overriding

(This one may not be possible to remedy) trulens_eval overrides a lot of methods
from other libraries when it wishes to track their usage.

## thread overriding

- "HACK002" -- We override `ThreadPoolExecutor`` in concurrent.futures.

### llama-index

- "HACK001" -- `trace_method` decorator does not preserve function signatures;
  we hack it so that it does. 

### langchain

- "HACK003" -- We override the base class of
  `langchain_core.runnables.config.ContextThreadPoolExecutor`

### pydantic

- endpoint extra args
- model_validate implementation in WithClassInfo
- 

### other

- "HACK004" -- Outdated, need investigation whether it can be removed.

