# Function Definitions

A feedback function scores the output of an LLM application by analyzing generated text as part of an LLM application (or a downstream model or application built on it). This guide provides details about the feedback functions that are implemented out of the box by TruLens. At the end of the guide, you can find additional information about how to create custom feedback functions.

See also: <https://www.trulens.org/trulens_eval/api/feedback/>

## Relevance

This evaluates the *relevance* of the LLM response to the given text by LLM prompting.

Relevance is currently only available with OpenAI ChatCompletion API.

TruLens offers two particular flavors of relevance:

1. *Prompt response relevance* is best for measuring the relationship of the final answer to the user inputed question. This flavor of relevance is particularly optimized for the following features:

    * Relevance requires adherence to the entire prompt.
    * Responses that don't provide a definitive answer can still be relevant
    * Admitting lack of knowledge and refusals are still relevant.
    * Feedback mechanism should differentiate between seeming and actual relevance.
    * Relevant but inconclusive statements should get increasingly high scores as they are more helpful for answering the query.

    You can read more information about the performance of prompt response relevance by viewing its [smoke test results](../answer_relevance_smoke_tests/).

2. *Question statement relevance*, sometimes known as context relevance, is best for measuring the relationship of a provided context to the user inputed question. This flavor of relevance is optimized for a slightly different set of features:
    * Relevance requires adherence to the entire query.
    * Long context with small relevant chunks are relevant.
    * Context that provides no answer can still be relevant.
    * Feedback mechanism should differentiate between seeming and actual relevance.
    * Relevant but inconclusive statements should get increasingly high scores as they are more helpful for answering the query.

    You can read more information about the performance of question statement relevance by viewing its [smoke test results](../context_relevance_smoke_tests/).

## Groundedness

Groundedness uses OpenAI LLMs or Huggingface NLI to attempt to check if an answer is grounded in its supplied contexts on a scale from 1 to 10. The information overlap or entailment between source and response is then measured, choosing the highest score between sources and then averaged and scaled from 0 to 1.

You can read about the performance of groundedness evaluations by viewing its [smoke test results](../groundedness_smoke_tests/).


## Sentiment

This evaluates the *positive sentiment* of either the prompt or response.

Sentiment is currently available to use with OpenAI, HuggingFace or Cohere as the model provider.

* The OpenAI sentiment feedback function prompts a Chat Completion model to rate the sentiment from 1 to 10, and then scales the response down to 0-1.
* The HuggingFace sentiment feedback function returns a raw score from 0 to 1.
* The Cohere sentiment feedback function uses the classification endpoint and a small set of examples stored in `feedback_prompts.py` to return either a 0 or a 1.

## Model Agreement

Model agreement uses OpenAI to attempt an honest answer at your prompt with system prompts for correctness, and then evaluates the agreement of your LLM response to this model on a scale from 1 to 10. The agreement with each honest bot is then averaged and scaled from 0 to 1.

## Language Match

This evaluates if the language of the prompt and response match.

Language match is currently only available to use with HuggingFace as the model provider. This feedback function returns a score in the range from 0 to 1, where 1 indicates match and 0 indicates mismatch.

## Toxicity

This evaluates the toxicity of the prompt or response.

Toxicity is currently only available to be used with HuggingFace, and uses a classification endpoint to return a score from 0 to 1. The feedback function is negated as not_toxicity, and returns a 1 if not toxic and a 0 if toxic.

## Moderation

The OpenAI Moderation API is made available for use as feedback functions. This includes hate, hate/threatening, self-harm, sexual, sexual/minors, violence, and violence/graphic. Each is negated (ex: not_hate) so that a 0 would indicate that the moderation rule is violated. These feedback functions return a score in the range 0 to 1.

## Stereotypes

This evaluates stereotypes using OpenAI LLMs to check if gender or race were assumed with no prior indication. This is rated on a scale from 1 to 10 where 10 being no new gender or race assumptions. A two indicates gender or race assumption with no indication, and a one indicates gender or race changes with prior indication that is different.

## Summarization

This evaluates summarization tasks using OpenAI LLMs to check how well a summarization hits upon main points. This is rated on a scale from 1 to 10 where 10 being all points are addressed. 

## Embeddings Distance

Given an embedder, as is typical in vector DBs, this evaluates the distance of the query and document embeddings. Currently supporting cosine distance, L1/Manhattan distance, and L2/Euclidean distance.