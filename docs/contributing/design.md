# 🧭 Design Goals and Principles

***Minimal time/effort-to-value*** If a user already has an LLM app coded in one of the
   supported libraries, provide immediate value with minimal additional effort required.

Currently to get going, a user needs to add 4 lines of Python:

```python
from trulens.dashboard import run_dashboard # line 1
from trulens.apps.langchain import TruChain # line 2
with TruChain(app): # 3
    app.invoke("some question") # doesn't count since they already had this

run_dashboard() # 4
```

3 of these lines are fixed so only #3 would vary in typical cases. From here
they can open the dashboard and inspect the recording of their app's invocation
including performance and cost statistics. This means TruLens must perform significant
processing under the hood to get that data. This is outlined primarily in
the [Instrumentation](#instrumentation) section below.

## Instrumentation

### App Data

We collect app components and parameters by walking over its structure and
producing a JSON representation with everything we deem relevant to track. The
function [jsonify][trulens.core.utils.json.jsonify] is the root of this process.

#### Class/system specific

##### Pydantic (LangChain)

Classes inheriting [BaseModel][pydantic.BaseModel] come with serialization
to/from JSON in the form of [model_dump][pydantic.BaseModel.model_dump] and
[model_validate][pydantic.BaseModel.model_validate]. We do not use the
serialization to JSON part of this capability as many *LangChain* components
fail serialization with a "will not serialize" message. However, we make
use of Pydantic `fields` to enumerate components of an object ourselves saving
us from having to filter out irrelevant internals that are not declared as
fields.

We make use of pydantic's deserialization, however, even for our own internal
structures (see `schema.py` for example).

##### dataclasses (no present users)

The built-in dataclasses package has similar functionality to Pydantic. We
use/serialize them using their field information.

##### dataclasses_json (LlamaIndex)

Placeholder. Currently no special handling is implemented.

##### Generic Python (portions of LlamaIndex and all else)

#### TruLens-specific Data

In addition to collecting app parameters, we also collect:

- (subset of components) App class information:

  - This allows us to deserialize some objects. Pydantic models can be
      deserialized once we know their class and fields, for example.
    - This information is also used to determine component types without having
      to deserialize them first.
    - See [Class][trulens.core.utils.pyschema.Class] for details.

### Functions/Methods

Methods and functions are instrumented by overwriting choice attributes in
various classes.

#### Class/system specific

##### Pydantic (LangChain)

Most if not all *LangChain* components use pydantic which imposes some
restrictions but also provides some utilities. Classes inheriting
[BaseModel][pydantic.BaseModel] do not allow defining new attributes but
existing attributes including those provided by pydantic itself can be
overwritten (like dict, for example). Presently, we override methods with
instrumented versions.

#### Alternatives

- `intercepts` package (see <https://github.com/dlshriver/intercepts>)

    Low level instrumentation of functions but is architecture and platform
    dependent with no darwin nor arm64 support as of June 07, 2023.

- `sys.setprofile` (see
  <https://docs.python.org/3/library/sys.html#sys.setprofile>)

    Might incur much overhead and all calls and other event types get
    intercepted and result in a callback.

- LangChain/LlamaIndex callbacks. Each of these packages come with some
  callback system that lets one get various intermediate app results. The
  drawbacks is the need to handle different callback systems for each system and
  potentially missing information not exposed by them.

- `wrapt` package (see <https://pypi.org/project/wrapt/>)

    This package only wraps functions or classes to resemble their originals.
    However, it doesn't help with wrapping existing methods in LangChain.
     We might be able to use it as part of our own wrapping scheme though.

### Calls

The instrumented versions of functions/methods record the inputs/outputs and
some additional data (see
[RecordAppCallMethod][trulens.core.schema.record.RecordAppCallMethod]). As more than
one instrumented call may take place as part of a app invocation, they are
collected and returned together in the `calls` field of
[Record][trulens.core.schema.record.Record].

Calls can be connected to the components containing the called method via the
`path` field of [RecordAppCallMethod][trulens.core.schema.record.RecordAppCallMethod].
This class also holds information about the instrumented method.

#### Call Data (Arguments/Returns)

The arguments to a call and its return are converted to JSON using the same
tools as App Data (see above).

#### Tricky

- The same method call with the same `path` may be recorded multiple times in a
  `Record` if the method makes use of multiple of its versions in the class
  hierarchy (i.e. an extended class calls its parents for part of its task). In
  these circumstances, the `method` field of
  [RecordAppCallMethod][trulens.core.schema.record.RecordAppCallMethod] will
  distinguish the different versions of the method.

- Thread-safety -- it is tricky to use global data to keep track of instrumented
  method calls in presence of multiple threads. For this reason we do not use
  global data and instead hide instrumenting data in the call stack frames of
  the instrumentation methods. See
  [get_all_local_in_call_stack][trulens.core.utils.python.get_all_local_in_call_stack].

- Generators and Awaitables -- If an instrumented call produces a generator or
  awaitable, we cannot produce the full record right away. We instead create a
  record with placeholder values for the yet-to-be produce pieces. We then
  instrument (i.e. replace them in the returned data) those pieces with (TODO
  generators) or awaitables that will update the record when they get eventually
  awaited (or generated).

#### Threads

Threads do not inherit call stacks from their creator. This is a problem due to
our reliance on info stored on the stack. Therefore we have a limitation:

- **Limitation**: Threads need to be started using the utility class
  [TP][trulens.core.utils.threading.TP] or
  [ThreadPoolExecutor][trulens.core.utils.threading.ThreadPoolExecutor] also
  defined in `utils/threading.py` in order for instrumented methods called in a
  thread to be tracked. As we rely on call stack for call instrumentation we
  need to preserve the stack before a thread start which Python does not do.

#### Async

Similar to threads, code run as part of a [asyncio.Task][] does not inherit
the stack of the creator. Our current solution instruments
[asyncio.new_event_loop][] to make sure all tasks that get created
in `async` track the stack of their creator. This is done in
[tru_new_event_loop][trulens.core.utils.python.tru_new_event_loop] . The
function [stack_with_tasks][trulens.core.utils.python.stack_with_tasks] is then
used to integrate this information with the normal caller stack when needed.
This may cause incompatibility issues when other tools use their own event loops
or interfere with this instrumentation in other ways. Note that some async
functions that appear not to involve [Task][asyncio.Task] do use tasks, such as
[gather][asyncio.gather].

- **Limitation**: [Task][asyncio.Task]s must be created via our `task_factory`
  as per
  [task_factory_with_stack][trulens.core.utils.python.task_factory_with_stack].
  This includes tasks created by function such as [asyncio.gather][]. This
  limitation is not expected to be a problem given our instrumentation except if
  other tools are used that modify `async` in some ways.

#### Limitations

- Threading and async limitations. See **Threads** and **Async** .

- If the same wrapped sub-app is called multiple times within a single call to
  the root app, the record of this execution will not be exact with regards to
  the path to the call information. All call paths will reference the last instrumented subapp
  (based on instrumentation order). For example, in a sequential app
  containing two of the same app, call records will be addressed to the second
  of the (same) apps and contain a list describing calls of both the first and
  second.

  TODO(piotrm): This might have been fixed. Check.

- Some apps cannot be serialized/JSONized. Sequential app is an example. This is
  a limitation of *LangChain* itself.

- Instrumentation relies on CPython specifics, making heavy use of the
  [inspect][] module which is not expected to work with other Python
  implementations.

#### Alternatives

- LangChain/llama_index callbacks. These provide information about component
  invocations but the drawbacks are need to cover disparate callback systems and
  possibly missing information not covered.

### Calls: Implementation Details

Our tracking of calls uses instrumentated versions of methods to manage the
recording of inputs/outputs. The instrumented methods must distinguish
themselves from invocations of apps that are being tracked from those not being
tracked, and of those that are tracked, where in the call stack a instrumented
method invocation is. To achieve this, we rely on inspecting the Python call
stack for specific frames:

- Prior frame -- Each instrumented call searches for the topmost instrumented
  call (except itself) in the stack to check its immediate caller (by immediate
  we mean only among instrumented methods) which forms the basis of the stack
  information recorded alongside the inputs/outputs.

#### Drawbacks

- Python call stacks are implementation dependent and we do not expect to
  operate on anything other than CPython.

- Python creates a fresh empty stack for each thread. Because of this, we need
  special handling of each thread created to make sure it keeps a hold of the
  stack prior to thread creation. Right now we do this in our threading utility
  class TP but a more complete solution may be the instrumentation of
  [threading.Thread][] class.

#### Alternatives

- [contextvars][] -- *LangChain* uses these to manage contexts such as those used
  for instrumenting/tracking LLM usage. These can be used to manage call stack
  information like we do. The drawback is that these are not threadsafe or at
  least need instrumenting thread creation. We have to do a similar thing by
  requiring threads created by our utility package which does stack management
  instead of contextvar management.

    NOTE(piotrm): it seems to be standard thing to do to copy the contextvars into
    new threads so it might be a better idea to use contextvars instead of stack
    inspection.
