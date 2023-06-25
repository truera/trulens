# Examples

## Contents

- `models`

    Examples using a variety of large language models from different sources.

    - `alpaca7b_local_llm.ipynb`

       Personal assistant with Alpaca7B running locally using HuggingFacePipeline's from_model_id.

- `trubot/`

    Examples based on a question-answering chain with context indexed from the
    TruEra website.

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

- `frameworks/`
    Collection of examples using different frameworks for constructing an LLM app.

    - `llama_index/`

        Examples using llama-index as a framework.

        - `llama_index_example.ipynb`

            Question-answering with a vector store of contexts loaded from a local
            set of files (`data` folder)

    - `langchain/`

        Examples using langchain as a framework.

        - `langchain_quickstart.ipynb`

            Question-answering with langchain

- `vector-dbs/`

    Collection of examples that makes use of vector databases for context
    retrieval in question answering.


    - `pinecone/`

      Examples that use llama-index as a framework and pinecone as the vector db.

        - `llama_index_pinecone_comparecontrast.ipynb`

            Using llama-index and pinecone to compare and contrast cities using their wikipedia articles.


- `app_with_human_feedback.py`

    Streamlit app with a langchain-based chat and the use of feedback functions
    based on user input.

- `feedback_functions.ipynb`

    A list of out of the box feedback functions, and how to contribute new ones.

- `logging.ipynb`

    Different ways to log your app with TruLens

- `quickstart.ipynb`

- `quickstart.py`

