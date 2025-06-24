# 🦙 LlamaIndex Integration

TruLens provides TruLlama, a deep integration with LlamaIndex to allow you to
inspect and evaluate the internals of your application built using LlamaIndex.
This is done through the instrumentation of key LlamaIndex classes and methods.
To see all classes and methods instrumented, see *Appendix: LlamaIndex
Instrumented Classes and Methods*.

## Example usage

Below is a quick example of usage. First, we'll create a standard LlamaIndex query engine from Paul Graham's Essay, *What I Worked On*:

!!! example "Create a LlamaIndex Query Engine"

    ```python
    from llama_index.core import VectorStoreIndex
    from llama_index.readers.web import SimpleWebPageReader

    documents = SimpleWebPageReader(html_to_text=True).load_data(
        ["http://paulgraham.com/worked.html"]
    )
    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine()
    ```

To instrument a LlamaIndex query engine, all that's required is to wrap it using TruLlama.

!!! example "Instrument a LlamaIndex Query Engine"

    ```python
    from trulens.apps.llamaindex import TruLlama

    tru_query_engine_recorder = TruLlama(query_engine)

    with tru_query_engine_recorder as recording:
        print(query_engine.query("What did the author do growing up?"))
    ```

To properly evaluate LLM apps, we often need to point our evaluation at an
internal step of our application, such as the retrieved context. Doing so allows
us to evaluate for metrics including context relevance and groundedness.

`TruLlama` supports `on_input`, `on_output`, and `on_context`, allowing you to easily evaluate the RAG triad.

Using `on_context` allows to access the retrieved text for evaluation via the source nodes of the LlamaIndex app.

!!! example "Evaluating retrieved context for LlamaIndex query engines"

    ```python
    import numpy as np
    from trulens.core import Feedback
    from trulens.providers.openai import OpenAI

    provider = OpenAI()

    context = TruLlama.select_context(query_engine)

    f_context_relevance = (
        Feedback(provider.context_relevance)
        .on_input()
        .on_context(collect_list=False)
        .aggregate(np.mean)
    )
    ```

You can find the full quickstart available here: [LlamaIndex Quickstart](../../getting_started/quickstarts/llama_index_quickstart.ipynb)

## Async Support
TruLlama also provides async support for LlamaIndex through the `aquery`,
`achat`, and `astream_chat` methods. This allows you to track and evaluate async
applications.

As an example, below is an LlamaIndex async chat engine (`achat`).

!!! example "Instrument an async LlamaIndex app"

    ```python
    from llama_index.core import VectorStoreIndex
    from llama_index.readers.web import SimpleWebPageReader
    from trulens.apps.llamaindex import TruLlama

    documents = SimpleWebPageReader(html_to_text=True).load_data(
        ["http://paulgraham.com/worked.html"]
    )
    index = VectorStoreIndex.from_documents(documents)

    chat_engine = index.as_chat_engine()

    tru_chat_recorder = TruLlama(chat_engine)

    with tru_chat_recorder as recording:
        llm_response_async = await chat_engine.achat(
            "What did the author do growing up?"
        )

    print(llm_response_async)
    ```

## Streaming Support

TruLlama also provides streaming support for LlamaIndex. This allows you to track and evaluate streaming applications.

As an example, below is an LlamaIndex query engine with streaming.

!!! example "Instrument an async LlamaIndex app"

    ```python
    from llama_index.core import VectorStoreIndex
    from llama_index.readers.web import SimpleWebPageReader

    documents = SimpleWebPageReader(html_to_text=True).load_data(
        ["http://paulgraham.com/worked.html"]
    )
    index = VectorStoreIndex.from_documents(documents)

    chat_engine = index.as_chat_engine(streaming=True)
    ```

As with other methods, simply wrap your streaming query engine with TruLlama and operate like before.

You can also print the response tokens as they are generated using the `response_gen` attribute.

!!! example "Instrument a streaming LlamaIndex app"

    ```python
    tru_chat_engine_recorder = TruLlama(chat_engine)

    with tru_chat_engine_recorder as recording:
        response = chat_engine.stream_chat("What did the author do growing up?")

    for c in response.response_gen:
        print(c)
    ```

For examples of using `TruLlama`, check out the [_TruLens_ Cookbook](../../cookbook/index.md)

## Appendix: LlamaIndex Instrumented Classes and Methods

The modules, classes, and methods that TruLens instruments can be retrieved from
the appropriate Instrument subclass.

!!! example

    ```python
    from trulens.apps.llamaindex import LlamaInstrument

    LlamaInstrument().print_instrumentation()
    ```

### Inspecting instrumentation

The specific objects (of the above classes) and methods instrumented for a
particular app can be inspected using the `App.print_instrumented` as
exemplified in the next cell. Unlike `Instrument.print_instrumentation`, this
function only shows specific objects and methods within an app that are actually instrumented.

!!! example

    ```python
    tru_chat_engine_recorder.print_instrumented()
    ```

### Instrumenting other classes/methods

Additional classes and methods can be instrumented by use of the
`trulens.core.otel.instrument` methods and decorators.
