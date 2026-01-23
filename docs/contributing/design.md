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

## OpenTelemetry-Based Instrumentation

As of TruLens 1.x, instrumentation is built on OpenTelemetry (OTEL). The OTEL
integration provides:

- **Standardized tracing**: Spans and traces follow OTEL conventions
- **Context propagation**: OTEL handles context across threads and async code
- **Flexible export**: Data can be exported to various backends via OTEL exporters

The OTEL-based instrumentation is in:

- `trulens.core.otel.instrument` - The `@instrument` decorator for marking methods
- `trulens.experimental.otel_tracing` - Session and exporter configuration
- `trulens.otel.semconv` - Semantic conventions for TruLens spans

The sections below describe the instrumentation implementation details, including
some legacy approaches that are being phased out.

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

- Thread-safety -- With OTEL, context propagation across threads is handled by
  OpenTelemetry's context API. Legacy code used call stack inspection (see
  [get_all_local_in_call_stack][trulens.core.utils.python.get_all_local_in_call_stack])
  but this is being phased out in favor of OTEL context.

- Generators and Awaitables -- If an instrumented call produces a generator or
  awaitable, we cannot produce the full record right away. We instead create a
  record with placeholder values for the yet-to-be produce pieces. We then
  instrument (i.e. replace them in the returned data) those pieces with (TODO
  generators) or awaitables that will update the record when they get eventually
  awaited (or generated).

#### Threads

With OTEL-based instrumentation, context propagation across threads is handled
by OpenTelemetry's context API, which properly propagates trace context.

Legacy instrumentation had this limitation:

- **Legacy Limitation**: Threads needed to be started using the utility class
  [TP][trulens.core.utils.threading.TP] or
  [ThreadPoolExecutor][trulens.core.utils.threading.ThreadPoolExecutor] in
  `utils/threading.py` for proper tracking. This is less critical with OTEL.

#### Async

With OTEL-based instrumentation, async context is handled by OpenTelemetry's
context API which integrates with Python's contextvars.

Legacy instrumentation used custom task factory instrumentation:

- **Legacy approach**: Instrumented [asyncio.new_event_loop][] via
  [tru_new_event_loop][trulens.core.utils.python.tru_new_event_loop] and used
  [task_factory_with_stack][trulens.core.utils.python.task_factory_with_stack]
  to track async task stacks. This is less critical with OTEL context propagation.

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

#### Current Approach (OTEL)

With OTEL-based instrumentation, call tracking uses OpenTelemetry spans:

- Each instrumented method creates a span with input/output attributes
- Parent-child relationships are managed by OTEL's context API
- Context propagates automatically across threads and async boundaries
- The `@instrument` decorator in `trulens.core.otel.instrument` handles span creation

#### Legacy Approach (Stack Inspection)

The legacy implementation relied on Python call stack inspection:

- Instrumented methods searched for the topmost instrumented call in the stack
- This required custom thread/async handling to preserve stack information
- Stack inspection is CPython-specific

This approach is phased out in favor of OTEL context propagation.
