# "Cannot find TruLens context" Warning/Error

```
Cannot find TruLens context. See
https://www.trulens.org/component_guides/other/no_context_warning for more information.
```

If you see this warning/error, _TruLens_ attempted to execute an instrumented
method in a context different from the one in which your app was instrumented. A
different context here means either a different `threading.Thread` or a
different `asyncio.Task`. While we include several remedies to this problem to
allow use of threaded and/or asynchronous apps, these remedies may not cover all cases. 
This document aims to help you resolve issues when your app or libraries aren't covered by our existing remedies.

## Threads

If using threads, use the replacement threading classes included in _TruLens_
that stand in place of Python classes:

- [trulens.core.utils.threading.Thread][trulens.core.utils.threading.Thread]
  instead of [threading.Thread][threading.Thread].

- [trulens.core.utils.threading.ThreadPoolExecutor][trulens.core.utils.threading.ThreadPoolExecutor]
  instead of
  [concurrent.futures.ThreadPoolExecutor][concurrent.futures.ThreadPoolExecutor].

You can also import either from their builtin locations as long as you import
_TruLens_ first.

Alternatively, use the utility methods in the [TP class][trulens.core.utils.threading.TP] such as
[submit][trulens.core.utils.threading.TP.submit].

Alternatively, use [Context.run][contextvars.Context.run] in your threads,
with the original target being the first argument to `run`:

```python
from contextvars import copy_context

# before:
Thread(target=your_thread_target, args=(yourargs, ...), kwargs=...)

# after:
Thread(target=copy_context().run, args=(your_thread_target, yourargs, ...), kwargs=...)
```

## Async Tasks

If using async tasks, ensure `Task` uses the default `copy_context` behavior. This only applies to Python >= 3.11:

!!! example

    ```python
    from contextvars import copy_context
    from asyncio import get_running_loop

    loop = get_running_loop()

    # before:
    task = loop.create_task(your_coroutine, ..., context=...)

    # after:
    task = loop.create_task(your_coroutine, ..., context=copy_context())
    # or:
    task = loop.create_task(your_coroutine, ...) # use default context behavior
    ```

Note: for Python < 3.11, `copy_context` is a fixed behavior and cannot be changed.

## Other issues

If you are still seeing the _Cannot find TruLens context_ warning and none of the solutions
above address the problem, please file a [GitHub Issue](https://github.com/truera/trulens/issues/new?template=bug-report.md) or 
add a new discussion on the [Snowflake Community Forums](https://snowflake.discourse.group/).
