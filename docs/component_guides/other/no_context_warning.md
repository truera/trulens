# "Cannot find TruLens Context" Warning

If you see the error or warning below:

```
Cannot find TruLens Context. See
https://www.trulens.org/component_guides/other/no_context_warning for more information.
```

it means that _TruLens_ attempted to execute an instrumented method in a context
different than the one in which your app was instrumented. A different context
here means either a different `threading.Thread` or a different `asyncio.Task`.
While we include several remedies to this problem to allow use of threaded or
asynchronous apps, these remedies may not cover all of the cases. This document
is here to help you fix the issue in case your app or the libraries you use was
not covered by our existing remedies.

## Threads

If using threads, use the replacement threading classes included in _TruLens_
that stand in place of python classes:

- [trulens.core.utils.threading.Thread][trulens.core.utils.threading.Thread]
  instead of [threading.Thread][threading.Thread].

- [trulens.core.utils.threading.ThreadPoolExecutor][trulens.core.utils.threading.ThreadPoolExecutor]
  instead of
  [concurrent.futures.ThreadPoolExecutor][concurrent.futures.ThreadPoolExecutor].

Alternatively, use the utility methods in the [TP
class][trulens.core.utils.threading.TP] such as
[submit][trulens.core.utils.threading.TP.submit].

Alternatively, target [Context.run][contextvars.Context.run] in your threads,
with the original target being the first argument to `run`:

```python
from contextvars import copy_context

# before:
Thread(target=your_thread_target, args=(yourargs, ...), kwargs=...)

# after
Thread(target=copy_context().run, args=(your_thread_target, yourargs, ...), kwargs=...)
```

## Async Tasks

If using async Tasks, make sure that the default `copy_context` behaviour of
`Task` is being used. This only applies to python >= 3.11:

```python
from contextvars import copy_context
from asyncio import get_running_loop

loop = get_running_loop()

# before:
task = loop.create_task(your_coroutine, ..., context=...)

# after:
task = loop.create_task(your_coroutine, ..., context=copy_context())
# or:
task = loop.create_task(your_coroutine, ...) # use default context behaviour
```

If you are using python prior to 3.11, `copy_context` is the fixed behaviour
which cannot be changed.

## Other issues

If you are still seeing the _Not Recording_ warning and none of the solutions
above address the problem, please post a [GitHub
issue](https://github.com/truera/trulens/issues) or a slack post on the
[AIQuality Forum](https://communityinviter.com/apps/aiqualityforum/josh).
