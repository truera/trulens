# Examples

## Contents

- **`models/`** -- Examples using a variety of large language models from different sources.

    - `alpaca7b_local_llm.ipynb` -- Personal assistant with Alpaca7B running locally using HuggingFacePipeline's from_model_id.

- **`frameworks/`** -- Collection of examples using different frameworks for constructing an LLM app.

    - `llama_index/`

        Examples using llama-index as a framework.

        - `llama_index_async.ipynb` -- How to use the async and streaming capabilities of Llama-Index and monitor the results using trulens.
     
        - `lllama_index_groundtruth.ipynb` -- Evaluate an Llama-Index application using ground truth.
        
        - `llama_index_example.ipynb` -- Question-answering with a in-memory vector store of contexts loaded from the web.
     
        - `llamaindex-subquestion-query.ipynb` -- Evaluate the impact of query planning and embedding choice.
     
        - `llamaindex-yelp-agent.ipynb` -- Evaluate hallucination of an Llama-Index data agent.

    - `langchain/` -- Examples using langchain as a framework.
 
        - `langchain_agents.ipynb` -- How to use the agents capability of Langchain and evaluate tool coverage using trulens.
        
        - `langchain_async.ipynb` -- How to use the streaming capability of Langchain and monitor the results using trulens.
     
        - `langchain_groundtruth.ipynb` -- Evaluate an LangChain application using ground truth.

        - `langchain_model_comparison.ipynb` -- Compare different models with TruLens in a langchain framework.

        - `langchain_quickstart.ipynb` -- Question-answering with LangChain.

        - `langchain_summarize.ipynb` -- A summarization model using langchain. This type of model does not
            take as input a piece of text but rather a set of documents.

- **`vector-dbs/`**

    Collection of examples that makes use of vector stores for context
    retrieval in question answering.

    - `pinecone/` -- Examples that use pinecone as the vector store.

        - `llama_index_pinecone_comparecontrast.ipynb` - Using llama-index and Pinecone to compare and contrast cities using their wikipedia articles.

        - `langchain-retrieval-augmentation-with-trulens.ipynb` - Iterating on an LLM application using LangChain and Pinecone for retrieval augmentation.
     
        - `constructing_optimal_pinecone.ipynb` - Experimenting with different Pinecone configurations.
     
    - `faiss/` -- Examples that use faiss as the vector store.
        
        - `langchain_faiss_example.ipynb` - Example using FAISS as the vector store.
     
- **`trubot/`** - End-to-end examples based on a question-answering chain with context indexed from the TruEra website.

    - `hnswlib_trubot/` -- local vector db data indexing the Truera website for
      trubot examples.

    - `App_TruBot.py` -- streamlit app to interact with trubot.

    - `trubot_example.ipynb` -- several variants of the question-answering chain
      addressing shortcomings of the original model.

    - `trubot_tests.ipynb` -- runs trubot on several example questions to
      quickly populate the dashboard with 4 model variants.

    - `trubot.py` -- trubot implementation as well as slack hooks if running as
      a slack app.

    - `webindex.ipynb` -- tools for indexing a website to produce a vector db
      for context.

- `app_with_human_feedback.py`-- Streamlit app with a langchain-based chat and the use of feedback functions
    based on user input.

- `feedback_functions.ipynb` -- A list of out of the box feedback functions.

- `logging.ipynb` -- Different ways to log your app with TruLens.

- `quickstart.ipynb` -- Quickstart with LangChain.

- `quickstart.py` -- .py version of quickstart with LangChain.
