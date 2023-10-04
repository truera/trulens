
# Use Cases

This section highlights different end-to-end use cases that TruLens can help with. For each use case, we not only motivate the use case but also discuss which components are most helpful for solving that use case.

## For Any Application

!!! info "[Model Selection](#)"
    Use TruLens to choose the most performant and efficient model for your application.

!!! info "[Moderation and Safety](#)"
    Monitor your LLM application responses against a set of moderation and safety checks.

!!! info "[Language Verification](#)"
    Verify your LLM application responds in the same language it is prompted.

## For Retrieval Augmented Generation (RAG)

!!! info "[Detect and Mitigate Hallucination](#)"
    Use groundedness feedback to ensure that your LLM responds using only the information retrieved from a verified knowledge source.

!!! info "[Improve Retrieval Quality](#)"
    Measure and identify ways to improve the quality of retrieval for your RAG.

!!! info "[Optimize App Configuration](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/expositional/vector-dbs/pinecone/pinecone_evals_build_better_rags.ipynb)"
    Iterate through a set of configuration options for your RAG including different metrics, parameters, models and more; find the most performant with TruLens.

## For LLM Agents

!!! info "[Validate LLM Agent Actions](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/expositional/frameworks/llama_index/llama_index_agents.ipynb)"
    Verify that your agent uses the intended tools and check it against business requirements.

!!! info "[Detect LLM Agent Tool Gaps/Drift](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/expositional/frameworks/langchain/langchain_agents.ipynb)"
    Identify when your LLM agent is missing the tools it needs to complete the tasks required.

## More Use Cases

!!! info "[Verify the Quality of Summarization](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/summarization_eval.ipynb)"
    Ensure that LLM summarizations contain the key points from source documents.

!!! info "[Async Evaluation](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/langchain_async.ipynb)"
    Evaluate your applications that leverage async mode.