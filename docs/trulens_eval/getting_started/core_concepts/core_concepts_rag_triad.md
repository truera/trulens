# The RAG Triad

RAGs have become the standard architecture for providing LLMs with context in order to avoid hallucinations. However even RAGs can suffer from hallucination, as is often the case when the retrieval fails to retrieve sufficient context or even retrieves irrelevant context that is then weaved into the LLM’s response.

TruEra has innovated the RAG triad to evaluate for hallucinations along each edge of the RAG architecture, shown below:

![RAG Triad](../assets/images/RAG_Triad.jpg)

The RAG triad is made up of 3 evaluations: context relevance, groundedness and answer relevance. Satisfactory evaluations on each provides us confidence that our LLM app is free form hallucination.

## Context Relevance

The first step of any RAG application is retrieval; to verify the quality of our retrieval, we want to make sure that each chunk of context is relevant to the input query. This is critical because this context will be used by the LLM to form an answer, so any irrelevant information in the context could be weaved into a hallucination. TruLens enables you to evaluate context relevance by using the structure of the serialized record.

## Groundedness

After the context is retrieved, it is then formed into an answer by an LLM. LLMs are often prone to stray from the facts provided, exaggerating or expanding to a correct-sounding answer. To verify the groundedness of our application, we can separate the response into individual claims and independently search for evidence that supports each within the retrieved context.

## Answer Relevance

Last, our response still needs to helpfully answer the original question. We can verify this by evaluating the relevance of the final response to the user input.

## Putting it together

By reaching satisfactory evaluations for this triad, we can make a nuanced statement about our application’s correctness; our application is verified to be hallucination free up to the limit of its knowledge base. In other words, if the vector database contains only accurate information, then the answers provided by the RAG are also accurate.

To see the RAG triad in action, check out the [TruLens Quickstart](https://www.trulens.org/trulens_eval/quickstart/)

