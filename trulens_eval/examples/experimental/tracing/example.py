import functools
from typing import Self, Type, Iterator, Generator, AsyncGenerator, Awaitable, AsyncIterator
import asyncio
import time
import requests

from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def instrument(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with tracer.start_as_current_span(func.__name__) as span:
            span.set_attribute("args", str(args))
            span.set_attribute("kwargs", str(kwargs))
            ret = func(*args, **kwargs)
            span.set_attribute("ret", str(ret))

            return ret

    return wrapper

@tracer.start_as_current_span("useautoinstrumented")
def useautoinstrumented():

    # Args to this method will be captured:
    requests.get("http://snowflake.com")

@tracer.start_as_current_span("innerfunction")
def innerfunction(a: int) -> int:

    current_span = trace.get_current_span()
    current_span.set_attribute("input_a", a)

    ret = a + 1

    current_span.set_attribute("ret", ret)

    return ret


@tracer.start_as_current_span("outerfunction")
def outerfunction(a: int) -> int:

    current_span = trace.get_current_span()
    current_span.set_attribute("input_a", a)

    ret = innerfunction(a) * 2

    current_span.set_attribute("ret", ret)

    return ret

# @tracer.start_as_current_span("somefunction")
@instrument
def somefunction(a: int) -> int:

    # current_span = trace.get_current_span()
    # current_span.set_attribute("input_a", a)

    return a + 1

# @tracer.start_as_current_span("somegenfunction")
@instrument
def somegenfunction(a: int) -> Iterator[int]:

    # TODO: How to establish a parent-child relationship between the trace
    # established by the decorator and the ones below? Instead creating a
    # context below:

    #with tracer.start_as_current_span("somegenfunction") as current_span:
    #    current_span.set_attribute("iteration_limit", a)
    for i in range(a):
        with tracer.start_as_current_span("somegenfunction_iterations") as iter_span:
            iter_span = trace.get_current_span()
            iter_span.set_attribute("iteration", i)
            time.sleep(1)
            yield i
    

# @tracer.start_as_current_span("someasyncfunction")
@instrument
async def someasyncfunction(a: int) -> int:
    # current_span = trace.get_current_span()
    # current_span.set_attribute("input_a", a)

    await asyncio.sleep(1)
    return a + 1

# @tracer.start_as_current_span("someasyncgenfunction")
async def someasyncgenfunction(a: int) -> AsyncIterator[int]:
    # Same problem as somegenfunction .

    #with tracer.start_as_current_span("someasyncgenfunction") as current_span:
    #    current_span.set_attribute("input_a", a)
    for i in range(a):
        with tracer.start_as_current_span("someasyncgenfunction_iterations") as iter_span:
            iter_span = trace.get_current_span()
            iter_span.set_attribute("iteration", i)
            await asyncio.sleep(1)
            yield i

class SomeClass(object):
    # @tracer.start_as_current_span("somemethod")
    @instrument
    def somemethod(self, a: int) -> int:
        # current_span = trace.get_current_span()
        # current_span.set_attribute("input_a", a)

        return a + 2
    
    #@tracer.start_as_current_span("somestaticmethod")
    @instrument
    @staticmethod
    def somestaticmethod(a: int) -> int:

        #current_span = trace.get_current_span()
        #current_span.set_attribute("input_a", a)

        return a + 3
    
    # @tracer.start_as_current_span("someclassmethod")
    @instrument
    @classmethod
    def someclassmethod(cls: Type[Self], a: int) -> int:

        current_span = trace.get_current_span()
        current_span.set_attribute("input_a", a)

        return a + 4

if False:
    somefunction(4)

    for a in somegenfunction(3):
        print(a)

    async def run_async():
        await someasyncfunction(2)
        async for a in someasyncgenfunction(4):
            print(a)

    asyncio.new_event_loop().run_until_complete(run_async())

if False:

    outerfunction(2)
    outerfunction(2)


if False:
    with tracer.start_as_current_span("root") as span:
        # All of these will have the same trace_id, that of span.

        innerfunction(1)
        innerfunction(1)

        outerfunction(2)
        outerfunction(2)

if False:
    with tracer.start_as_current_span("root") as root_span:
        # All of these will have the same trace_id, that of root_span.

        # tracer2 has no impact on the innerfunction as they were decorated with
        # tracer.
        tracer2 = trace.get_tracer(__name__) # NOTE: arg is NOT tracer identifier

        innerfunction(1)

        with tracer2.start_as_current_span("subroot") as subroot_span:
            innerfunction(1)

if True:
    useautoinstrumented()