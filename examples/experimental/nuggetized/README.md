# Nuggetized Feedback
Information nuggets were introduced long ago as part of an early [TREC Track](https://trec.nist.gov/pubs/trec12/papers/QA.OVERVIEW.pdf) focused on question answering. This track defined the category as: "An information nugget was defined as a fact for which the assessor could make a binary decision as to whether a response contained the nugget. At the end of this step, the assessor decided which nuggets were vital—nuggets that must appear in a definition for that definition to be good". In other words, an information nugget is a fact or statement which when presented with an answer, it is immediately visible if this "nugget" is present.

In the 2024 and beyond RAG tracks, nuggets became the default way of understanding response qualities of RAG systems as it could verify the presence of indivisible facts or statements and verify the presence.

## Nuggetization Process. (Original)
As shown in [The Great Nugget Recall](https://arxiv.org/pdf/2504.15068) the nuggetization process is a simple multi step process which goes as follows:
1. First, given a piece of context and a user query or intent, extract the relevant nuggets. This runs iteratively until no new nuggets are created or a max threshold is met (30 in TREC RAG)
2. Given this set of nuggets, they are then scored for importance relative to query needs. The labels are either "vital" or "ok". "ok" is related but not needed while "vital" is, as implied, vital to the query.
3. Once there are importance scores, the final step is to measure if nuggets are supported by the passage which they come from. Nuggets/context relationship can be "support" implying the nugget is fully supported by the passage, "not_support", and "partial_support", which imply what their names indicate.

## Nuggets in TruLens
TruLens is built around the notion of the RAG triad, which decomposes RAG answer quality into context relevance, groundedness, and answer relevance. Context relevance measures the relevance of the selected context to the original query, groundedness measures how grounded in the context the answer is, and answer relevance measures how relevant the answer is to the question. This approach is simple and effective but lacks the expressivity of the nugget approach.

To make nuggets fit within the RAG triad we treat the process of nuggetization as a wrapper to the existing feedback process. Context relevance is unchanged so existing passages are evaluated on relevance with direct comparing the context passage with the query. Groundness and answer relevance are changed as the existing RAG triad is provided with nuggetized information. This means that Nuggetized RAG triad behaves as follows:

1. First, given a piece of context and a user query or intent, extract the relevant nuggets. This runs iteratively until no new nuggets are created or a max threshold is met (30 in TREC RAG). This is done based on the full generated answer.
2. Given this set of nuggets, they are then scored for importance relative to query needs. The labels are either "vital" or "ok". "ok" is related but not needed while "vital" is, as implied, vital to the query.
3. Nuggets are passed to the existing RAG triad and evaluated independently.
4. RAG triad results are combined and weighted based on nugget importance.

This process is demonstrated by running the script below. Be sure to set up your OPEN_AI_API_KEY before proceeding.

[nuggetized_feedback.py](../../../src/feedback/trulens/feedback/nuggetized_feedback.py)

## Expected Output
======================================================================
RAG RESPONSE EVALUATION
======================================================================

Question: What are the key features of Python programming language?

Context (first 150 chars):
    Python is a high-level, interpreted programming language known for its
    simplicity and readability. It was created by Guido van Rossum and fi...

Answer (first 150 chars):
    Python's key features include its simple and readable syntax that emphasizes
    code clarity. It supports multiple programming paradigms like o...
Temperature parameter is not supported for reasoning model gpt-5-nano. Removing temperature parameter.
Temperature parameter is not supported for reasoning model gpt-5-nano. Removing temperature parameter.
Temperature parameter is not supported for reasoning model gpt-5-nano. Removing temperature parameter.
Temperature parameter is not supported for reasoning model gpt-5-nano. Removing temperature parameter.

======================================================================
EVALUATION RESULTS
======================================================================

📊 TRADITIONAL EVALUATION:
   Groundedness: 86.7%
   Relevance:    100.0%

🔬 NUGGETIZED EVALUATION:
   Groundedness: 81.2%
   Relevance:    35.2%

   Evaluated 9 nuggets:
   1. 'Simple, readable syntax emphasizing code clarity...'
      Importance: Vital, Score: 66.7%
   2. 'Supports multiple paradigms: object-oriented and f...'
      Importance: Vital, Score: 100.0%
   3. 'Dynamic typing...'
      Importance: Vital, Score: 100.0%
   ... and 6 more nuggets

📈 COMPARISON:
   Groundedness difference: -5.4%
   Relevance difference:    -64.8%

💡 INTERPRETATION:
   Both methods show similar groundedness scores.
   Nuggetized evaluation shows lower relevance,
   indicating varied relevance across answer components.

✅ Evaluation completed successfully!
