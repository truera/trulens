"""Optional import messages for the backwards compat trulens_eval package."""

from trulens.core.utils.imports import format_import_errors

# Optional sub-packages:
REQUIREMENT_FEEDBACK = format_import_errors(
    "trulens-feedback", purpose="evaluating feedback functions"
)

# Other optionals:
REQUIREMENT_GROUNDEDNESS = format_import_errors(
    "nltk", purpose="using some groundedness feedback functions"
)

REQUIREMENT_PINECONE = format_import_errors(
    # package name is "pinecone-client" but module is "pinecone"
    ["pinecone-client", "langchain_community"],
    purpose="running TruBot",
)

REQUIREMENT_SKLEARN = format_import_errors(
    "scikit-learn", purpose="using embedding vector distances"
)
REQUIREMENT_BERT_SCORE = format_import_errors(
    "bert-score", purpose="measuring BERT Score"
)
REQUIREMENT_EVALUATE = format_import_errors(
    "evaluate", purpose="using certain metrics"
)
REQUIREMENT_NOTEBOOK = format_import_errors(
    ["ipython", "ipywidgets"], purpose="using TruLens-Eval in a notebook"
)
REQUIREMENT_OPENAI = format_import_errors(
    ["openai", "langchain_community"], purpose="using OpenAI models"
)

# Optional provider types:
REQUIREMENT_PROVIDER_BEDROCK = format_import_errors(
    "trulens-providers-bedrock", purpose="evaluating feedback using Bedrock"
)
REQUIREMENT_PROVIDER_CORTEX = format_import_errors(
    "trulens-providers-cortex", purpose="evaluating feedback using Cortex"
)
REQUIREMENT_PROVIDER_HUGGINGFACE = format_import_errors(
    "trulens-providers-huggingface",
    purpose="evaluating feedback using Huggingface",
)
REQUIREMENT_PROVIDER_LANGCHAIN = format_import_errors(
    "trulens-providers-langchain", purpose="evaluating feedback using LangChain"
)
REQUIREMENT_PROVIDER_LITELLM = format_import_errors(
    "trulens-providers-litellm", purpose="evaluating feedback using LiteLLM"
)
REQUIREMENT_PROVIDER_OPENAI = format_import_errors(
    "trulens-providers-openai", purpose="evaluating feedback using OpenAI"
)
