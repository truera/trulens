# 🦙 LlamaIndex Integration

TruLens provides TruLlama, a deep integration with LlamaIndex to allow you to
inspect and evaluate the internals of your application built using LlamaIndex.
This is done through the instrumentation of key LlamaIndex classes and methods.
To see all classes and methods instrumented, see *Appendix: LlamaIndex
Instrumented Classes and Methods*.

In addition to the default instrumentation, TruLlama exposes the
*select_context* and *select_source_nodes* methods for evaluations that require
access to retrieved context or source nodes. Exposing these methods bypasses the
need to know the json structure of your app ahead of time, and makes your
evaluations reusable across different apps.

## Example usage

Below is a quick example of usage. First, we'll create a standard LlamaIndex query engine from Paul Graham's Essay, *What I Worked On*

!!! example "Create a Llama-Index Query Engine"

    ```python
    from llama_index.core import VectorStoreIndex
    from llama_index.readers.web import SimpleWebPageReader

    documents = SimpleWebPageReader(html_to_text=True).load_data(
        ["http://paulgraham.com/worked.html"]
    )
    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine()
    ```

To instrument an Llama-Index query engine, all that's required is to wrap it using TruLlama.

!!! example "Instrument a Llama-Index Query Engine"

    ```python
    from trulens.apps.llamaindex import TruLlama

    tru_query_engine_recorder = TruLlama(query_engine)

    with tru_query_engine_recorder as recording:
        print(query_engine.query("What did the author do growing up?"))
    ```

To properly evaluate LLM apps we often need to point our evaluation at an
internal step of our application, such as the retrieved context. Doing so allows
us to evaluate for metrics including context relevance and groundedness.

For LlamaIndex applications where the source nodes are used, `select_context`
can be used to access the retrieved text for evaluation.

!!! example "Evaluating retrieved context for Llama-Index query engines"

    ```python
    import numpy as np
    from trulens.core import Feedback
    from trulens.providers.openai import OpenAI

    provider = OpenAI()

    context = TruLlama.select_context(query_engine)

    f_context_relevance = (
        Feedback(provider.context_relevance)
        .on_input()
        .on(context)
        .aggregate(np.mean)
    )
    ```

You can find the full quickstart available here: [Llama-Index Quickstart](../../getting_started/quickstarts/llama_index_quickstart.ipynb)

## Async Support
TruLlama also provides async support for LlamaIndex through the `aquery`,
`achat`, and `astream_chat` methods. This allows you to track and evaluate async
applications.

As an example, below is an LlamaIndex async chat engine (`achat`).

!!! example "Instrument an async Llama-Index app"

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

!!! example "Instrument an async Llama-Index app"

    ```python
    from llama_index.core import VectorStoreIndex
    from llama_index.readers.web import SimpleWebPageReader

    documents = SimpleWebPageReader(html_to_text=True).load_data(
        ["http://paulgraham.com/worked.html"]
    )
    index = VectorStoreIndex.from_documents(documents)

    chat_engine = index.as_chat_engine(streaming=True)
    ```

Just like with other methods, just wrap your streaming query engine with TruLlama and operate like before.

You can also print the response tokens as they are generated using the `response_gen` attribute.

!!! example "Instrument a streaming Llama-Index app"

    ```python
    tru_chat_engine_recorder = TruLlama(chat_engine)

    with tru_chat_engine_recorder as recording:
        response = chat_engine.stream_chat("What did the author do growing up?")

    for c in response.response_gen:
        print(c)
    ```

For examples of using `TruLlama`, check out the [_TruLens_ Cookbook](../../cookbook/index.md)

## Appendix: LlamaIndex Instrumented Classes and Methods

The modules, classes, and methods that trulens instruments can be retrieved from
the appropriate Instrument subclass.

!!! example

    ```python
    from trulens.apps.llamaindex import LlamaInstrument

    LlamaInstrument().print_instrumentation()
    ```

### Instrumenting other classes/methods.
Additional classes and methods can be instrumented by use of the
`trulens.core.instruments.Instrument` methods and decorators. Examples of
such usage can be found in the custom app used in the `custom_example.ipynb`
notebook which can be found in
`examples/expositional/end2end_apps/custom_app/custom_app.py`. More
information about these decorators can be found in the
`docs/trulens/tracking/instrumentation/index.ipynb` notebook.

### Inspecting instrumentation
The specific objects (of the above classes) and methods instrumented for a
particular app can be inspected using the `App.print_instrumented` as
exemplified in the next cell. Unlike `Instrument.print_instrumentation`, this
function only shows what in an app was actually instrumented.

!!! example

    ```python
    tru_chat_engine_recorder.print_instrumented()
    ```
