Initialize feedback database with this before running apps:

```
import sqlite3

# Connect to the database
conn = sqlite3.connect('llm_quality.db')
c = conn.cursor()


# Commit changes and close the connection
conn.commit()
conn.close()
```

You can then run:

`streamlit run gpt3_streamlit.py` - template app where llm is called and feedback is logged.

`streamlit run feedback_viewer.py` - dashboard for feedback
# Additional Contents

- `tru.py` - add_data implementation for collecting feedback
- `tru_feedback.py` - implementation of feedback functions
- `requirements.txt` - pip requirements
- `Makefile` - targets for useful things
- `slackbot.py` - slack bot implementing TruBot
- `keys.py` - imports api secrets from your local .env
- `webscrape` - folder for storing cached web scape metrials used for creating a document database for bot
- `webindex.ipynb` - web scraper and document index uploader (to pinecone)
- `piotr-workshop.ipynb` - unsorted things Piotr is trying out
- `tru_chain.py` -- langchain.Chain wrapper
- `test_tru_chain.py` -- examples/tests for wrapper
- `benchmark.py` -- Module to enable benchmarking feedback functions
- `benchmarking.ipynb` -- Benchmarking stock feedback functions