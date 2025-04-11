# Instrumentation Overview

TruLens is a framework that helps you instrument and evaluate LLM apps including
RAGs and agents.

Because TruLens is tech-agnostic, we offer a few different tools for
instrumentation.

* TruCustomApp gives you the most power to instrument a custom LLM app, and
  provides the `instrument` method.
* TruBasicApp is a simple interface to capture the input and output of a basic
  LLM app.
* TruChain instruments LangChain apps. [Read more](langchain.md).
* TruLlama instruments LlamaIndex apps. [Read more](llama_index.md).
* TruRails instruments NVIDIA NeMo Guardrails apps. [Read more](nemo.md).

In any framework you can track (and evaluate) the inputs, outputs and
instrumented internals, along with a wide variety of usage metrics and metadata,
detailed below:

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

### App Metadata

* App ID (app_id) - user supplied string or automatically generated hash
* Tags (tags) - user supplied string
* Model metadata - user supplied json

### Record Metadata

* Record ID (record_id) - automatically generated, track individual application
  calls
* Timestamp (ts) - automatically tracked, the timestamp of the application call
* Latency (latency) - the difference between the application call start and end
  time.

!!! example "Using `@instrument`"

    ```python
    from trulens.apps.custom import instrument

    class RAG_from_scratch:
        @instrument
        def retrieve(self, query: str) -> list:
            """
            Retrieve relevant text from vector store.
            """

        @instrument
        def generate_completion(self, query: str, context_str: list) -> str:
            """
            Generate answer from context.
            """

        @instrument
        def query(self, query: str) -> str:
            """
            Retrieve relevant text given a query, and then generate an answer from the context.
            """

    ```

In cases you do not have access to a class to make the necessary decorations for
tracking, you can instead use one of the static methods of instrument, for
example, the alternative for making sure the custom retriever gets instrumented
is via `instrument.method`. See a usage example below:

!!! example "Using `instrument.method`"

    ```python
    from trulens.apps.custom import instrument
    from somepackage.from custom_retriever import CustomRetriever

    instrument.method(CustomRetriever, "retrieve_chunks")

    # ... rest of the custom class follows ...
    ```

Read more about instrumenting [custom class applications][trulens.apps.custom.TruCustomApp]

## Tracking input-output applications

For basic tracking of inputs and outputs, `TruBasicApp` can be used for instrumentation.

Any text-to-text application can be simply wrapped with `TruBasicApp`, and then recorded as a context manager.

!!! example "Using `TruBasicApp` to log text to text apps"

    ```python
    from trulens.apps.basic import TruBasicApp

    def custom_application(prompt: str) -> str:
        return "a response"

    basic_app_recorder = TruBasicApp(
        custom_application, app_id="Custom Application v1"
    )

    with basic_app_recorder as recording:
        basic_app_recorder.app("What is the phone number for HR?")
    ```

For frameworks with deep integrations, TruLens can expose additional internals
of the application for tracking. See [TruChain][trulens.apps.langchain] and [TruLlama][trulens.apps.llamaindex] for more details.
