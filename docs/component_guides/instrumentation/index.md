# Instrumentation Overview

TruLens is a framework designed to help you instrument and evaluate LLM applications, including RAGs and agents. TruLens instrumentation is [OpenTelemetry](https://opentelemetry.io/) compatible, allowing you to interoperate with other observability systems.

!!! note

    OpenTelemetry tracing is enabled by default. To disable it, set the environment variable `TRULENS_OTEL_TRACING` to "0" or "false".

This instrumentation capability allows you to track the entire execution flow of your app, including inputs, outputs, internal operations, and performance metrics.

## Why Instrument?

Instrumentation serves two key purposes:

1. **Observability**: Track the execution flow of your application
2. **Evaluation**: Select instrumented attributes for evaluation with feedback functions

The attributes you capture during instrumentation become available for evaluation. For example, if you instrument the retrieved contexts in a RAG application, you can then evaluate those contexts for relevance or groundedness. See [Feedback Selectors](../evaluation/feedback_selectors/index.md) to learn how to evaluate instrumented attributes.

## Instrumenting Applications with `@instrument`

For applications that you can edit the source code, TruLens provides a framework-agnostic `instrument` decorator to capture the information from decorated functions. More specifically, adding the `instrument()` decorator will allow TruLens to log the function signature as span attributes.

Consider the following instrumented class method, `retrieve_contexts`:

!!! example

    ```python
    from typing import List

    from opentelemetry import trace
    from trulens.core.otel.instrument import instrument


    class MyRAG:
        @instrument()
        def retrieve_contexts(
            self, query: str
        ) -> List[str]:
            """This function has no custom attributes."""
            return ["context 1", "context 2"]
    ```

In the example above, the `query` argument is logged as `ai.observability.call.kwargs.query` and the function return value is logged as `ai.observability.call.return`.

## Instrumenting custom attributes

To capture the values from the function signature as specific span attributes, you can pass in a dictionary to the `attributes` parameter of the `@instrument` decorator where the keys are function arguments or `return` for the return value.

Adding custom attributes in this way does not capture any additional information, however it can allow you to capture these span attributes in a way that is semantically meaningful to your application, or to adhere to existing standards.

!!! example

    ```python
        @instrument(
        attributes={
            "custom_attr__query": "query",
            "custom_attr__results": "return",
        }
    )
    def retrieve_contexts_with_function_signature_attributes(
        self, query: str
    ) -> List[str]:
        return ["context 3", "context 4"]
    ```

!!! tip "Evaluating Custom Attributes"

    Custom attributes you instrument can be selected for evaluation using feedback functions. See [Selecting Spans for Evaluation](../evaluation/feedback_selectors/selecting_components.md) to learn how to evaluate these instrumented attributes.

## Instrumenting custom attributes with _TruLens_ semantic conventions

`instrument()` also allows you to annotate methods with _TruLens_ semantic conventions that add meaning to the instrumented attributes. You can read more about the _TruLens_ [semantic conventions](https://www.trulens.org/otel/semantic_conventions/) which lay out how to emit spans.

In the example below, you can see how we use _TruLens_ semantic conventions to instrument the span types `RETRIEVAL`, `GENERATION` and `RECORD_ROOT`.

In the `retrieve` method, we also associate the `query` argument with the span attribute `RETRIEVAL.QUERY_TEXT`, and the method's `return` with `RETRIEVAL.RETRIEVED_CONTEXT`. We follow a similar process for the `query` method.

In addition to using the `attributes` arg to pass in a dictionary of span attributes, we the example below also shows how to set the `span_type` of instrumented methods.

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

## Manipulating custom attributes

In some cases, you may want to manipulate information from the function signature before instrumenting. For example, if the retrieved context is buried inside of nested dict.

The `@instrument` decorator provides powerful flexibility through lambda functions in the `attributes` parameter. Instead of simple static mappings, you can use lambda functions to dynamically compute custom attributes based on the function's execution context.

When you provide a lambda function to the `attributes` parameter, you gain access to:

1. `ret` - The function's return value (useful for extracting data from complex responses)
2. `exception` - Any exception that was thrown during execution (None if successful)
3. `*args` - All positional arguments passed to the function
4. `**kwargs` - All keyword arguments (positional args are also included here by name)

The example below demonstrates advanced attribute manipulation:

* `custom_attr__retrieved_texts`: Uses the `ret` parameter to extract the "text" values from each dictionary in the returned list, creating a clean list of just the text content for instrumentation.
* `custom_attr__uppercased_query`: Uses the `kwargs` parameter to access the input query and transform it (uppercase) before storing as an attribute.

The lambda function dynamically processes both the function's return value and input parameters to create meaningful instrumentation data.

!!! example

    ```python
    from trulens.core.otel.instrument import instrument
    from trulens.otel.semconv.trace import SpanAttributes

        @instrument(
            attributes=lambda ret, exception, *args, **kwargs: {
                SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: [doc["text"] for doc in ret],
                SpanAttributes.RETRIEVAL.QUERY_TEXT: kwargs["query"].upper()
            }
        )
        def retrieve_contexts(
            self, query: str
        ) -> List[Dict[str, str]]:
            return [
                {"text": "context 5", "source": "doc1.pdf"},
                {"text": "context 6", "source": "doc2.pdf"}
            ]
    ```

!!! tip "Evaluating Manipulated Attributes"

    Attributes you transform using lambda functions are fully available for evaluation. This is particularly useful when your data requires preprocessing before evaluation. See [Selecting Spans for Evaluation](../evaluation/feedback_selectors/selecting_components.md) to learn how to evaluate these instrumented attributes.

## Instrumenting Common App Frameworks

In cases where you are leveraging frameworks like `LangChain`, `LangGraph` and `LlamaIndex`, TruLens instruments the framework for you. To take advantage of this instrumentation, you can simply use `TruChain` ([Read more](langchain.md)) for `LangChain`, `TruGraph` ([Read more](langgraph.md)) for `LangGraph`, or `TruLlama` ([Read more](llama_index.md)) for `LlamaIndex` to wrap your application.

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

    === "_LangGraph_"

        ```python
        from trulens.apps.langgraph import TruGraph

        graph = graph_builder.compile()

        tru_recorder = TruGraph(
            graph,
            app_name="LangGraph Agent",
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
