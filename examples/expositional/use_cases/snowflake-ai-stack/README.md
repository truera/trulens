# ❄️ Snowflake AI Stack

This project gives you a starting point for building RAG with the Snowflake AI stack, so you can choose which componenents fit for your RAG system and see how the full picture works together.

All together, the app has the following features:

- **Flexible knowledge base**: Use any data source, just change the loader.
- **Performant Retrieval**: Uses Arctic Embed for high performance similarity search.
- **No Hallucination** Uses TruLens LLM Judge for filtering retrieved context before generation.
- **Observability**: Uses TruLens for tracing and evaluating application runs, which are displayed in interface.
- **User Interface**: An interactive app, built with Streamlit, including streaming for generation and multi-turn chat.

## 🏃 Setup Instructions

1) Clone the repo & navigate to the example
git clone https://github.com/truera/trulens.git
cd trulens/examples/expositional/use_cases/snowflake-ai-stack

2) Install the dependencies
pip install -r requirements.txt

For this demo, we will run two versions of the app.

3) Add required keys to `.sh` files.

4) RAG with no context filters:

`bash run_app1.sh`

5) RAG with context filters:

`bash run_app2_filters.sh`


## 🔨 Component Snowflake OSS Libraries

This repository gives you a starting point for building production-ready RAG systems using the Snowflake AI stack.
- [Arctic Embed](https://github.com/Snowflake-Labs/arctic-embed) (embedding & retrieval)
- [TruLens](https://www.trulens.org/) (Tracing, Evals & Guardrails)
- [Streamlit](https://streamlit.io/) (User Interface)

## 🛣️ Roadmap
- Clarifying Questions using [Arctic Agentic RAG](https://github.com/Snowflake-Labs/Arctic_Agentic_RAG) (RAG Framework)

## 🤝 Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.
