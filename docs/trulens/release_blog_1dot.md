# Moving to TruLens v1: Reliable and Modular Logging and Evaluation

It has always been our goal to make it easy to build trustworthy LLM applications. Since we launched last May, the package has grown up before our eyes, morphing from a hacked-together addition to an existing project (trulens-explain) to a thriving, agnostic standard for tracking and evaluating LLM apps. Along the way, we’ve experienced growing pains and discovered inefficiencies in the way TruLens was built. We’ve also heard that the reasons people use TruLens today are diverse, and many of its use cases do not require its full footprint. Today we’re announcing an extensive re-architecture of TruLens that aims to give developers a stable, modular platform for logging and evaluation they can rely on.

## **TLDR;**

### Split off trulens-eval from trulens-explain

Split off trulens-eval from trulens-explain, and let trulens-eval take over the trulens package name. TruLens-eval is now renamed to TruLens and sits at the root of the TruLens repo, while TruLens-explain has been moved to its own repository, and is installable at trulens-explain.

![TruLens 1.0 Release Graphics](../assets/images/trulens_1_release_graphic_split.png)

### Separate TruLens-Eval into different trulens packages

trulens-core holds core abstractions for database operations, app instrumentation, guardrails and evaluation
trulens-dashboard gives you the required capabilities to run and operate the TruLens dashboard
trulens-instrument-*/ describes a set of third-party integrations for instrumentation and logging, including our popular TruChain and TruLlama offerings that seamlessly instrument LangChain and Llama-Index apps. Each of these integrations can be installed separately as a standalone package, and include trulens-instrument-langchain, trulens-instrument-llamaindex and trulens-instrument-nemo.
trulens-feedback gives you access to out of the box feedback functions required for running feedback functions. Feedback function implementations must be combined with a selected provider integration.
trulens-providers-*/ describes a set of third party integrations for running feedback functions. Today, we offer an extensive set of integrations that allow you to run feedback functions on top of virtually any LLM. These integrations can be installed as standalone packages, and include: trulens-providers-openai, trulens-providers-huggingface, trulens-providers-litellm, trulens-providers-langchain, trulens-providers-bedrock, trulens-providers-cortex.

![TruLens 1.0 Release Graphics](../assets/images/trulens_1_release_graphic_modular.png)

## **Motivation**

The driving motivation behind these changes are twofold:

* Minimize the overhead required for TruLens developers to use the capabilities they need
* Make it easy to understand what code and set of dependencies is needed

## **Versioning and Backwards Compatibility**

Today, we’re releasing trulens, trulens-core, trulens-dashboard, trulens-feedback, trulens-providers packages and trulens-instrument packages at v1.0. We will not make breaking changes in the future without bumping the major version.

The base install of trulens will install trulens-core, trulens-feedback and trulens-dashboard making it easy for developers to try TruLens.

trulens-eval starting 1.0.0 will consist solely of backwards compatibility modules that alias trulens-* packages while emitting deprecation warnings. trulens-eval version will track trulens-core version for 6 weeks (Until October 10, 2024) after which the deprecation warnings will be replaced by error messages instructing the user to update to trulens.

During the depreciation period, trulens-eval will track the trulens version and contain aliases for non-private API elements while giving deprecation warnings upon usage. After the depreciation period, the latest version of trulens_eval will refuse to install with a note that it has been deprecated.

Read more about the [TruLens deprecation policy](./contributing/policies.md).

Along with this change, we’ve also included a [migration guide](./guides/trulens_eval/trulens_eval_migration.md) for moving to TruLens v1.

Please give us feedback on GitHub by creating issues and starting discussions.

To see the core re-architecture changes in action, we've included some usage examples below:

!!! example "Log and Instrument LLM Apps"

    === "python"

        ```bash
        pip install trulens-core
        ```

        ```python
        from trulens.core.app.custom import instrument

        class CustomApp:

            def __init__(self):
                self.retriever = CustomRetriever()
                self.llm = CustomLLM()
                self.template = CustomTemplate(
                    "The answer to {question} is {answer}"
                )

            @instrument
            def retrieve_chunks(self, data):
                return self.retriever.retrieve_chunks(data)

            @instrument
            def respond_to_query(self, input):
                chunks = self.retrieve_chunks(input)
                answer = self.llm.generate(",".join(chunks))
                output = self.template.fill(question=input, answer=answer)

                return output

        ca = CustomApp()
        ```

    === "Langchain"

        ```bash
        pip install trulens-instrument-langchain
        ```

        ```python
        from langchain import hub
        from langchain.chat_models import ChatOpenAI
        from langchain.schema import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        retriever = vectorstore.as_retriever()

        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        from trulens.instrument.langchain import TruChain

        # Wrap application
        tru_recorder = TruChain(
            chain,
            app_id='Chain1_ChatApplication'
        )

        # Record application runs
        with tru_recorder as recording:
            chain("What is langchain?")
        ```

    === "Llama-Index"

        ```bash
        pip install trulens-core trulens-instrument-llamaindex
        ```

        ```python
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

        from trulens.instrument.llamaindex import TruLlama
        from trulens.core import Feedback

        tru_recorder = TruLlama(query_engine,
            app_id='LlamaIndex_App1')

        with tru_recorder as recording:
            query_engine.query("What is llama index?")
        ```

!!! example "Run Feedback Functions with different LLMs"

    === "OpenAI LLMs"

        ```bash
        pip install trulens-core  trulens-providers-openai
        ```

        ```python
        from trulens.providers.openai import OpenAI
        from trulens.core import Feedback
        import numpy as np

        provider = OpenAI()

        # Use feedback
        f_context_relevance = (
            Feedback(provider.context_relevance_with_context_reasons)
            .on_input()
            .on(context)  # Refers to context defined from `select_context`
            .aggregate(np.mean)
        )
        ```

    === "Local LLMs with Ollama"

        ```bash
        pip install trulens-core trulens-providers-litellm
        ```

        ```python
        from trulens.providers.litellm import LiteLLM
        from trulens.core import Feedback
        import numpy as np

        provider = LiteLLM(
            model_engine="ollama/llama3.1:8b", api_base="http://localhost:11434"
        )

        # Use feedback
        f_context_relevance = (
            Feedback(provider.context_relevance_with_context_reasons)
            .on_input()
            .on(context)  # Refers to context defined from `select_context`
            .aggregate(np.mean)
        )
        ```

    === "Classification Models with Huggingface API"

        ```bash
        pip install trulens-core trulens-providers-huggingface
        ```

        ```python
        from trulens.core import Feedback
        from trulens.core import Select
        from trulens.providers.huggingface import Huggingface

        # Define a remote Huggingface groundedness feedback function
        remote_provider = Huggingface()
        f_remote_groundedness = (
            Feedback(
                remote_provider.groundedness_measure_with_nli,
                name="[Remote] Groundedness",
            )
            .on(Select.RecordCalls.retrieve.rets.collect())
            .on_output()
        )
        ```

        === "Local Classification Models with Huggingface"

        ```bash
        pip install trulens-core trulens-providers-huggingface
        ```

        ```python
        from trulens.core import Feedback
        from trulens.core import Select
        from trulens.providers.huggingface import HuggingfaceLocal

        # Define a local Huggingface groundedness feedback function
        local_provider = HuggingfaceLocal()
        f_local_groundedness = (
            Feedback(
                local_provider.groundedness_measure_with_nli,
                name="[Local] Groundedness",
            )
            .on(Select.RecordCalls.retrieve.rets.collect())
            .on_output()
        )
        ```

!!! example "Run the TruLens dashboard:"

    ```bash
    pip install trulens-dashboard
    ```

    ```python
    from trulens.core import Tru
    from trulens.dashboard import run_dashboard

    tru = Tru()

    run_dashboard(tru)
    ```

## **TruLens Sessions**

In TruLens, we have long had the `Tru()` class, a singleton that sets the logging configuration. If instantiated with no arguments, traces and evaluations are logged to a local sqlite file. In the case you don’t deliberately instantiate Tru(), traces and evaluations are still logged to a local sqlite file.

In v1, we are renaming `Tru` to `TruSession`, to represent a session for logging TruLens traces and evaluations.

You can see how to start a TruLens session logging to a postgres database below:

!!! example "Start a TruLens Session"

    ```python
    from trulens.core import TruSession
    from trulens.core.database.connector import DefaultDBConnector

    connector = DefaultDBConnector(database_url="postgresql://trulensuser:password@localhost/trulens")
    session = TruSession(connector=connector)
    ```

## **Up-leveled Experiment Tracking**

In v1, we’re also introducing new ways to track experiments with app_name and app_version. These new required arguments replace app_id to give you a more dynamic way to track app versions.

In our suggested workflow, app_name represents an objective you’re building your LLM app to solve. All apps with the same app_name should be directly comparable with each other. Then app_version can be used to track each experiment. This should be changed each time you change your application configuration. To more explicitly track the changes to individual configurations and semantic names for versions - you can still use app metadata and tags!

!!! example "Track Experiments"

    ```python
    tru_rag = TruCustomApp(
    rag,
    app_name="RAG",
    app_vesion="v1",
    tags="prototype",
    metadata=metadata={
                "top_k": top_k,
                "chunk_size": chunk_size,
            }
    )
    ```

To bring these changes to life, we've also added new filters to the Leaderboard and Evaluations pages. These filters give you the power to focus in on particular apps and versons, or even slice to apps with a specific tag or metadata.

## **First-class support for Ground Truth Evaluation**

Along with the high level changes in TruLens v1, ground truth can now be persisted in SQL-compatible datastores and loaded on demand as pandas dataframe objects in memory as required. By enabling the persistence of ground truth data, you can now run evaluations without loading ground truth data into memory.

## **New Conceptual Guide and TruLens Cookbook**

On the top-level of TruLens docs, we previously had separated out Evaluation, Evaluation Benchmarks, Tracking and Guardrails. These are now combined to form the new Conceptual Guide.

We also pulled in our extensive GitHub examples library directly into docs. This should make it easier for you to learn about all of the different ways to get started using TruLens. If you’re new to the examples library, the examples are organized into two categories: quickstart and expositional.

Quickstart examples are tested with every release, and show core functionality of the TruLens package.
Expositional examples focus on using TruLens with different data sources, models, frameworks and more. They are generally sorted by the type of integration you’re looking to use. For example, if you want to learn how to run feedback functions with a new LLM, you should check out expositional/models. Alternatively, if you want to learn how TruLens can instrument a new framework, you should check out expositional/frameworks.

## Conclusion

Ready to get started with the v1 stable release of TruLens? Check out our migration guide, or just jump in to the quickstart!
