# ❄️ Open Source RAG Stack
How do you build a highly performant, production-ready RAG system all in open source?

This project gives you a starting point for building RAG with the open source snowflake stack, so you can choose which componenents fit for your RAG system and see how the full picture works together.

All together, the app has the following features:

- **Flexible knowledge base**: Use any website as the knowledge base of your RAG.
- **In Memory Vector Store**: Manages the storage, indexing and retrieval of document vectors in memory.
- **Performant Retrieval**: Uses Arctic Embed and MMR for high performance search.
- **No Hallucination** Uses TruLens LLM Judge for filtering retrieved context before generation.
- **Low Latency Generation**: Inference Optimized Model used for generation
- **Observability**: Uses TruLens for tracing and evaluating application runs, which are displayed in interface.
- **User Interface**: An interactive app, built with Streamlit, including streaming for generation and multi-turn chat.

## 👀 See it in action

https://github.com/user-attachments/assets/3689c142-4ceb-4658-83a3-e24950af7fb3

## 🏃 Setup Instructions

1) Clone the repo
git clone https://github.com/Snowflake-Labs/oss-rag-stack.git

2) Install the dependencies
pip install -r requirements.txt

For this demo, we will run three versions of the app.

(3) RAG with no context filters on the base vLLM backend:

`bash run_app1_nofilters.sh`

(4) RAG with context filters on the base vLLM backend:

`bash run_app2_filters.sh`

(5) RAG with context filters on our optimized vLLM backend:

`bash run_app3_arctic.sh`

## 🔨 Component Snowflake OSS Libraries

This repository gives you a starting point for building production-ready RAG systems using the Snowflake OSS stack.
- [Arctic Embed](https://github.com/Snowflake-Labs/arctic-embed) (embedding & retrieval)
- Fast LLM Inference from [Arctic Training](https://arctictraining.readthedocs.io/en/latest/) (Inference Optimized LLMs)
- [TruLens](https://www.trulens.org/) (Tracing, Evals & Guardrails)
- [Streamlit](https://streamlit.io/) (User Interface)

## More about inference optimization.

In the backend, we have set up Llama-3.3-70B-Instruct running on (1) barebones vLLM, and (2) vLLM with SwiftKV and Speculative decoding. All you need to do is run the streamlit apps to point to them. To do so, you must be first connected to the Dev VPN.

## 🛣️ Roadmap
- Clarifying Questions using [Arctic Agentic RAG](https://github.com/Snowflake-Labs/Arctic_Agentic_RAG) (RAG Framework)
- Answer questions across multiple websites at once

## 🤝 Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.
