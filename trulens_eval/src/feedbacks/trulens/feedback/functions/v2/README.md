This readme presents the goals, planning, and design of feedback function reorganization work.

# Goals

Abstraction of feedback functions to expose most salient aspects to user while hiding implementation details.

# Plan

Abstractions organized into several layers with the current/old abstraction mostly occupying the last 1.5 layers. Ideally a user will not have to deal with anything beyond the first 2 layers and usually just the first layer unless they need to explore things like reasoning/chain of thought behind feedback results.

# First level abstraction

Highest level abstraction of feedback should be free of implementation details and instead focused on the meaning of the feedback itself, with possible examples, links to readings, benefits, drawbacks, etc. The mkdocs generated from this level would serve as good sources of information regarding higher-level feedback concepts.

Examples of other tools with similar abstraction are:

- [Langchain eval criteria](https://python.langchain.com/docs/guides/evaluation/string/criteria_eval_chain) . No organization is involved but implementation is somewhat abstracted.
- [OpenAI moderation](https://platform.openai.com/docs/guides/moderation)  . A minimal level organization is present there. Restricted to concepts related to OpenAI usage policies. Specific moderation model involved/exposed but typical usage ignores it.

Exposed in this layer:

## Organization/hierarchy of feedback functions

Some initial thoughts on root organization:

QUESTION: should positive/negative desirability to be part of this initial abstraction.

Feedback
- NaturalLanguage
  - Syntax
    - Language Match
    - SyntacticGroundTruth
  - Semantics
    - GroundTruth
    - Conciseness
    - Coherence
    - Relevance
      - QuestionStatementRelevance
      - PromptResponseRelevance
    - Groundedness
    - Sentiment
    - Helpfulness
    - Controversiality
    - Moderation
      - Stereotypes
      - Legality
        - Criminality
      - Harmfulness
      - Toxicity
      - Maliciousness
      - Disinformation
      - Hate
        - Misogyny
        - HateThreatening

## Docs/Refs/Citations

Any helpful references or pointers should be included here. Public datasets that include examples would be most helpful here. Can include samples from those in the docs as well.

## Examples

Examples of what the feedback function should produce. This part can interact with few-shot-classification in the lower level of abstraction described later. Examples are also critical in distinguishing the many related feedback functions that exist in trulens presently.

## Prompts

While specific to completion models, prompts are important to a user's understanding of what a feedback function measures so at least generic parts of prompts can be included in this first layer of abstraction. Prompts can be used in the lower-level implementation abstraction described below.

## Aggregate Docstring

Given all of the above, the user should be able to get a good picture of the feedback function by reading its docstring or some aggregated doc that combines all of the user-facing info listed above. In this manner, various text presently in notebook cells would be converted into the feedback docs associated with this first level of abstraction.

# Second level

Second level of abstraction exposes feedback output types and presence/absence/support of additional user-helpful aspects of feedback

- Binary outputs and interpretation of the two outputs.
- Digital (1 through 10) and interpretation if needed.
- Explanations of feedback function outputs.
- COT explanations.

Construction of feedbacks to include explanations based on feedbacks from level 1 is included here.

# Third level

Third level exposes models but tries to disentangle the service/api they are hosted on. Here we also distinguish model type in terms of classification vs. completion.

## Deriving

The ability to create a classifier from a completion model via a prompt and a few examples is to be exposed here.

# Fourth level

Fourth level is the most of the present layer with accounting of what service/api and completion model is to be used for the implementation.
