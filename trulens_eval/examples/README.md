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

- `vector-dbs/`

    Collection of examples that makes use of vector databases for context
    retrieval in question answering.

    - `llama_index/`

        - `llama_example.ipynb`

        - `llama_quickstart.py`

            Question-answering with a vector store of contexts loaded from a local
            set of files (`data` folder)

        - `essay.py`

            Llama_index starter example from
            https://gpt-index.readthedocs.io/en/latest/getting_started/starter_example.html
            .    

    - `llama_pinecone/`

        - `llama_pinecone.ipynb`

- `app_with_human_feedback.py`

    Streamlit app with a langchain-based chat and the use of feedback functions
    based on user input.

- `feedback_functions.ipynb`

- `logging.ipynb`

- `quickstart.ipynb`

- `quickstart.py`

