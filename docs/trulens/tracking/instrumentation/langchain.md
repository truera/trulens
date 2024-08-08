# 🦜️🔗 _LangChain_ Integration

TruLens provides TruChain, a deep integration with _LangChain_ to allow you to
inspect and evaluate the internals of your application built using _LangChain_.
This is done through the instrumentation of key _LangChain_ classes. To see a list
of classes instrumented, see *Appendix: Instrumented _LangChain_ Classes and
Methods*.

In addition to the default instrumentation, TruChain exposes the
*select_context* method for evaluations that require access to retrieved
context. Exposing *select_context* bypasses the need to know the json structure
of your app ahead of time, and makes your evaluations re-usable across different
apps.

## Example Usage

To demonstrate usage, we'll create a standard RAG defined with Langchain Expression Language (LCEL).

First, this requires loading data into a vector store.

!!! example "Create a RAG with LCEL"

    ```python
    import bs4
    from langchain.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain import hub
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents, embeddings)

    retriever = vectorstore.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    ```

To instrument an LLM chain, all that's required is to wrap it using TruChain.

!!! example "Instrument with `TruChain`"

    ```python
    from trulens.instrument.langchain import TruChain

    # instrument with TruChain
    tru_recorder = TruChain(rag_chain)
    ```

To properly evaluate LLM apps we often need to point our evaluation at an
internal step of our application, such as the retrieved context. Doing so allows
us to evaluate for metrics including context relevance and groundedness.

For LangChain applications where the BaseRetriever is used, `select_context` can
be used to access the retrieved text for evaluation.

!!! example "Evaluating retrieved context in Langchain"

    ```python
    import numpy as np
    from trulens.core import Feedback
    from trulens.providers.openai import OpenAI

    provider = OpenAI()

    context = TruChain.select_context(rag_chain)

    f_context_relevance = (
        Feedback(provider.context_relevance)
        .on_input()
        .on(context)
        .aggregate(np.mean)
    )
    ```


You can find the full quickstart available here: [LangChain Quickstart](trulens.org/trulens/getting_started/quickstarts/langchain_quickstart/)

## Async Support

TruChain also provides async support for _LangChain_ through the `acall` method. This allows you to track and evaluate async and streaming _LangChain_ applications.

As an example, below is an LLM chain set up with an async callback.

!!! example "Create an async chain with LCEL"

    ```python

    from langchain.callbacks import AsyncIteratorCallbackHandler
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from trulens.instrument.langchain import TruChain

    # Set up an async callback.
    callback = AsyncIteratorCallbackHandler()

    # Setup a simple question/answer chain with streaming ChatOpenAI.
    prompt = PromptTemplate.from_template(
        "Honestly answer this question: {question}."
    )
    llm = ChatOpenAI(
        temperature=0.0,
        streaming=True,  # important
        callbacks=[callback],
    )
    async_chain = LLMChain(llm=llm, prompt=prompt)
    ```

Once you have created the async LLM chain you can instrument it just as before.

!!! example "Instrument async apps with `TruChain`"

    ```python
    async_tc_recorder = TruChain(async_chain)

    with async_tc_recorder as recording:
        await async_chain.ainvoke(
            input=dict(question="What is 1+2? Explain your answer.")
        )
    ```

For more usage examples, check out the [LangChain cookbook](https://github.com/truera/trulens/tree/main/examples/expositional/frameworks/langchain).

## Appendix: Instrumented LangChain Classes and Methods

The modules, classes, and methods that trulens instruments can be retrieved from
the appropriate Instrument subclass.

!!! example "Instrument async apps with `TruChain`"

    ```python
    from trulens.instrument.langchain import LangChainInstrument

    LangChainInstrument().print_instrumentation()
    ```

### Instrumenting other classes/methods

Additional classes and methods can be instrumented by use of the
`trulens.core.instruments.Instrument` methods and decorators. Examples of
such usage can be found in the custom app used in the `custom_example.ipynb`
notebook which can be found in
`examples/expositional/end2end_apps/custom_app/custom_app.py`. More
information about these decorators can be found in the
`docs/tracking/instrumentation/index.ipynb` notebook.

### Inspecting instrumentation

The specific objects (of the above classes) and methods instrumented for a
particular app can be inspected using the `App.print_instrumented` as
exemplified in the next cell. Unlike `Instrument.print_instrumentation`, this
function only shows what in an app was actually instrumented.

!!! example "Print instrumented methods"

    ```python
    async_tc_recorder.print_instrumented()
    ```
