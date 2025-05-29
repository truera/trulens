export OPENAI_API_KEY="sk-proj-..."
export OPENAI_BASE_URL=https://gtc-demo-1.metac2.us-west-2.aws-dev.app.snowflake.com/v1 # TODO: change to ArcticInference
export REASONING_MODEL_NAME="llama-3.3-70b"
export LLM_MODEL_NAME="llama-3.3-70b"
export TAVILY_API_KEY="tvly-dev-..."
export TRULENS_OTEL_TRACING="1"
export STREAMLIT_SERVER_PORT=9001
export INCLUDE_EVAL_NODES="1"
streamlit run app.py
