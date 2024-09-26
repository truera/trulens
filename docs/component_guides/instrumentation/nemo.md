# _NeMo Guardrails_ Integration

TruLens provides TruRails, an integration with _NeMo Guardrails_ apps to allow you to
inspect and evaluate the internals of your application built using _NeMo Guardrails_.
This is done through the instrumentation of key _NeMo Guardrails_ classes. To see a list
of classes instrumented, see *Appendix: Instrumented Nemo Classes and
Methods*.

In addition to the default instrumentation, TruRails exposes the
*select_context* method for evaluations that require access to retrieved
context. Exposing *select_context* bypasses the need to know the json structure
of your app ahead of time, and makes your evaluations reusable across different
apps.

## Example Usage

Below is a quick example of usage. First, we'll create a standard Nemo app.

!!! example "Create a NeMo app"

    ```python
    %%writefile config.yaml
    # Adapted from NeMo-Guardrails/nemoguardrails/examples/bots/abc/config.yml
    instructions:
    - type: general
        content: |
        Below is a conversation between a user and a bot called the trulens Bot.
        The bot is designed to answer questions about the trulens python library.
        The bot is knowledgeable about python.
        If the bot does not know the answer to a question, it truthfully says it does not know.

    sample_conversation: |
    user "Hi there. Can you help me with some questions I have about trulens?"
        express greeting and ask for assistance
    bot express greeting and confirm and offer assistance
        "Hi there! I'm here to help answer any questions you may have about the trulens. What would you like to know?"

    models:
    - type: main
        engine: openai
        model: gpt-3.5-turbo-instruct

    %%writefile config.co
    # Adapted from NeMo-Guardrails/tests/test_configs/with_kb_openai_embeddings/config.co
    define user ask capabilities
    "What can you do?"
    "What can you help me with?"
    "tell me what you can do"
    "tell me about you"

    define bot inform capabilities
    "I am an AI bot that helps answer questions about trulens."

    define flow
    user ask capabilities
    bot inform capabilities

    # Create a small knowledge base from the root README file.

    ! mkdir -p kb
    ! cp ../../../../README.md kb

    from nemoguardrails import LLMRails
    from nemoguardrails import RailsConfig

    config = RailsConfig.from_path(".")
    rails = LLMRails(config)
    ```

To instrument an LLM chain, all that's required is to wrap it using TruChain.

!!! example "Instrument a NeMo app"

    ```python
    from trulens.apps.nemo import TruRails

    # instrument with TruRails
    tru_recorder = TruRails(
        rails,
        app_id="my first trurails app",  # optional
    )
    ```

To properly evaluate LLM apps we often need to point our evaluation at an
internal step of our application, such as the retrieved context. Doing so allows
us to evaluate for metrics including context relevance and groundedness.

For Nemo applications with a knowledge base, `select_context` can
be used to access the retrieved text for evaluation.

!!! example "Instrument a NeMo app"

    ```python
    import numpy as np
    from trulens.core import Feedback
    from trulens.providers.openai import OpenAI

    provider = OpenAI()

    context = TruRails.select_context(rails)

    f_context_relevance = (
        Feedback(provider.context_relevance)
        .on_input()
        .on(context)
        .aggregate(np.mean)
    )
    ```

For examples of using `TruRails`, check out the [_TruLens_ Cookbook](../../cookbook/index.md)

## Appendix: Instrumented Nemo Classes and Methods

The modules, classes, and methods that trulens instruments can be retrieved from
the appropriate Instrument subclass.

!!! example

    ```python
    from trulens.apps.nemo import RailsInstrument

    RailsInstrument().print_instrumentation()
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
    tru_recorder.print_instrumented()
    ```
