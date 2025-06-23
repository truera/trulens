# Instrumentation Overview

TruLens is a framework designed to help you instrument and evaluate LLM applications, including RAGs and agents. TruLens instrumentation is [OpenTelemetry](https://opentelemetry.io/) compatible, allowing you to interoperate with other observability systems.

!!! note

    To turn on OpenTelemetry tracing, set the environment variable `TRULENS_OTEL_TRACING` to "1".

This instrumentation capability allows you to track the entire execution flow of your app, including inputs, outputs, internal operations, and performance metrics.

## Instrumenting Applications with `@instrument`

For applications that you can edit the source code, TruLens provides a framework-agnostic `instrument` decorator to annotate methods with their span type and attributes. TruLens [semantic conventions](https://www.trulens.org/otel/semantic_conventions/) lay out how to emit spans.

In the example below, you can see how we use TruLens semantic conventions to instrument the span types `RETRIEVAL`, `GENERATION` and `RECORD_ROOT`.

In the `retrieve` method, we also associate the `query` argument with the span attribute `RETRIEVAL.QUERY_TEXT`, and the method's `return` with `RETRIEVAL.RETRIEVED_CONTEXT`. We follow a similar process for the `query` method.

!!! example

    ```python
    from trulens.core.otel.instrument import instrument
    from trulens.otel.semconv.trace import SpanAttributes

    class RAG:
        @instrument(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            attributes={
                SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
                SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
            },
        )
        def retrieve(self, query: str) -> list:
            """
            Retrieve relevant text from vector store.
            """

        @instrument(span_type=SpanAttributes.SpanType.GENERATION)
        def generate_completion(self, query: str, context_str: list) -> str:
            """
            Generate answer from context.
            """

        @instrument(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            attributes={
                SpanAttributes.RECORD_ROOT.INPUT: "query",
                SpanAttributes.RECORD_ROOT.OUTPUT: "return",
            },
        )
        def query(self, query: str) -> str:
            """
            Retrieve relevant text given a query, and then generate an answer from the context.
            """

    ```

## Instrumenting Common App Frameworks

In cases where you are leveraging frameworks like `Langchain` or `LlamaIndex`, TruLens instruments the framework for you. To take advantage of this instrumentation, you can simply use `TruChain` ([Read more](langchain.md))for `Langchain` apps or `TruLlama` ([Read more](llama_index.md))for `LlamaIndex` apps to wrap your application.

!!! example

    === "_LangChain_"

        ```python
        from trulens.apps.langchain import TruChain

        rag_chain = (
            {"context": filtered_retriever
            | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        tru_recorder = TruChain(
            rag_chain,
            app_name="ChatApplication",
            app_version="Base"
        )
        ```

    === "_LlamaIndex_"

        ```python
        from trulens.apps.llamaindex import TruLlama

        query_engine = index.as_query_engine(similarity_top_k=3)

        tru_query_engine_recorder = TruLlama(
            query_engine,
            app_name="LlamaIndex_App",
            app_version="base"
        )
        ```

## Instrumenting Input/Output Apps

TruBasicApp is a simple interface to capture the input and output of a basic LLM app. Using TruBasicApp requires no direct instrumentation, simply wrapping your app with the `TruBasicApp` class.

!!! example

    ```python
    from trulens.apps.basic import TruBasicApp

    def chat(prompt):
    return (
        client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        .choices[0]
        .message.content
    )

    tru_recorder = TruBasicApp(
        chat,
        app_name="base"
    )
    ```

### Instrumenting apps via `instrument_method()`

In cases when you do not have access to directly modify the source code of a class
(e.g. adding decorations for tracking), you can use static instrumentation methods
instead: for example, the alternative for making sure the custom retriever gets
instrumented is via `instrument_method`. See a usage example below:

!!! example "Using `instrument.method`"

    ```python
    from trulens.core.otel.instrument import instrument_method
    from somepackage.custom_retriever import CustomRetriever

    instrument_method(
        cls = CustomRetriever,
        method_name = "retrieve",
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        }
        )

    # ... rest of the custom class follows ...
    ```

## Tracking Usage Metrics

TruLens tracks the following usage metrics by capturing them from LLM spans.

### Usage Metrics

* Number of requests (n_requests)
* Number of successful ones (n_successful_requests)
* Number of class scores retrieved (n_classes)
* Total tokens processed (n_tokens)
* In streaming mode, number of chunks produced (n_stream_chunks)
* Number of prompt tokens supplied (n_prompt_tokens)
* Number of completion tokens generated (n_completion_tokens)
* Cost in USD (cost)

Read more about Usage Tracking in [Cost API Reference][trulens.core.schema.base.Cost].
