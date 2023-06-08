"""
# Model Instrumentation

## Designs and Choices

### Model Data

We collect model components and parameters by walking over its structure and
producing a json reprensentation with everything we deem relevant to track. The
function `util.py:jsonify` is the root of this process.

#### class/system specific

##### pydantic (langchain)

Classes inheriting `pydantic.BaseModel` come with serialization to/from json in
the form of `BaseModel.dict` and `BaseModel.parse`. We do not use the
serialization to json part of this capability as a lot of langchain components
are tripped to fail it with a "will not serialize" message. However, we use make
use of pydantic `fields` to enumerate components of an object ourselves saving
us from having to filter out irrelevant internals.

We make use of pydantic's deserialization, however, even for our own internal
structures (see `schema.py`).

##### dataclasses (no present users)

The built-in dataclasses package has similar functionality to pydantic but we
presently do not handle it as we have no use cases.

##### dataclasses_json (llama_index)

Work in progress.

##### generic python (portions of llama_index and all else)

#### TruLens-specific Data

In addition to collecting model parameters, we also collect:

- (subset of components) Model class information:

    - This allows us to deserialize some objects. Pydantic models can be
      deserialized once we know their class and fields, for example.
    - This information is also used to determine component types without having
      to deserialize them first. 
    - See `schema.py:Class` for details.

#### Tricky

#### Limitations

### Functions/Methods

Methods and functions are instrumented by overwriting choice attributes in
various classes. 

#### class/system specific

##### pydantic (langchain)

Most if not all langchain components use pydantic which imposes some
restrictions but also provides some utilities. Classes inheriting
`pydantic.BaseModel` do not allow defining new attributes but existing
attributes including those provided by pydantic itself can be overwritten (like
dict, for examople). Presently, we override methods with instrumented versions.

#### Alternatives

- `intercepts` package (see https://github.com/dlshriver/intercepts)

    Low level instrumentation of functions but is architecture and platform
    dependent with no darwin nor arm64 support as of June 07, 2023.

- `sys.setprofile` (see
  https://docs.python.org/3/library/sys.html#sys.setprofile)

    Might incur much overhead and all calls and other event types get
 
    intercepted and result in a callback.

### Tricky

- 

### Calls

The instrumented versions of functions/methods record the inputs/outputs and
some additional data (see `schema.py:RecordChainCall`). As more then one
instrumented call may take place as part of a model invokation, they are
collected and returned together in the `calls` field of `schema.py:Record`.

Calls can be connected to the components containing the called method via the
`path` field of `schema.py:RecordChainCallMethod`. This class also holds
information about the instrumented method.

#### Call Data (Arguments/Returns)

The arguments to a call and its return are converted to json using the same
tools as Model Data (see above).

#### Tricky

- The same method call with the same `path` may be recorded multiple times in a
  `Record` if the method makes use of multiple of its versions in the class
  hierarchy (i.e. an extended class calls its parents for part of its task). In
  these circumstances, the `method` field of `RecordChainCallMethod` will
  distinguish the different versions of the method.

- Thread-safety -- it is tricky to use global data to keep track of instrumented
  method calls in presence of multiple threads. For this reason we do not use
  global data and instead hide instrumenting data in the call stack frames of
  the instrumentation methods. See `util.py:get_local_in_call_stack.py`.

#### Limitations

- Threads need to be started using the utility class TP in order for
  instrumented methods called in a thread to be tracked. As we rely on call
  stack for call instrumentation we need to preserve the stack before a thread
  start which python does not do.  See `util.py:TP._thread_starter`.

"""

from datetime import datetime
from inspect import BoundArguments, signature
import os
from pprint import PrettyPrinter
import logging
from typing import Any, Callable, Dict, Sequence, Union
import threading as th

from pydantic import BaseModel
from trulens_eval.trulens_eval.schema import LangChainModel, MethodIdent, RecordChainCall, RecordChainCallMethod
from trulens_eval.trulens_eval.tru_chain import TruChain

from trulens_eval.trulens_eval.tru_db import Query

import langchain
from trulens_eval.trulens_eval.tru_feedback import Feedback
from trulens_eval.trulens_eval.util import get_local_in_call_stack, jsonify, noserio

logger = logging.getLogger(__name__)

class Instrument(object):
    pass

class LlamaInstrument(Instrument):
    pass

class LangChainInstrument(Instrument):
    pass