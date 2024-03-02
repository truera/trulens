# Honest, Harmless and Helpful Evaluations

TruLens adapts ‘**honest**, **harmless**, **helpful**’ as desirable criteria for
LLM apps from Anthropic. These criteria are simple and memorable, and seem to
capture the majority of what we want from an AI system, such as an LLM app.

## TruLens Implementation

To accomplish these evaluations we've built out a suite of evaluations (feedback
functions) in TruLens that fall into each category, shown below. These feedback
funcitons provide a starting point for ensuring your LLM app is performant and
aligned.

![Honest Harmless Helpful Evals](../../../assets/images/Honest_Harmless_Helpful_Evals.jpg)

## Honest

- At its most basic level, the AI applications should give accurate information.

- It should have access too, retrieve and reliably use the information needed to
  answer questions it is intended for.

**See honest evaluations in action:**

- [Building and Evaluating a prototype RAG](1_rag_prototype.ipynb)

- [Reducing Hallucination for RAGs](2_honest_rag.ipynb)

## Harmless

- The AI should not be offensive or discriminatory, either directly or through
  subtext or bias.

- When asked to aid in a dangerous act (e.g. building a bomb), the AI should
  politely refuse. Ideally the AI will recognize disguised attempts to solicit
  help for nefarious purposes.

- To the best of its abilities, the AI should recognize when it may be providing
  very sensitive or consequential advice and act with appropriate modesty and
  care.

- What behaviors are considered harmful and to what degree will vary across
  people and cultures. It will also be context-dependent, i.e. it will depend on
  the nature of the use.

**See harmless evaluations in action:**

- [Harmless Evaluation for LLM apps](3_harmless_eval.ipynb)

- [Improving Harmlessness for LLM apps](4_harmless_rag.ipynb)

## Helpful

- The AI should make a clear attempt to perform the task or answer the question
  posed (as long as this isn’t harmful). It should do this as concisely and
  efficiently as possible.

- Last, AI should answer questions in the same language they are posed, and
  respond in a helpful tone.

**See helpful evaluations in action:**

- [Helpful Evaluation for LLM apps](5_helpful_eval.ipynb)