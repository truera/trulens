# 📋 Feedback Template Reference

Feedback evaluation templates live in
`trulens.feedback.templates` and define the prompt / criteria scaffolding
that feedback providers use to build the system and user prompts for each
LLM-based evaluation. They are grouped into four domain modules — RAG,
safety, quality, and agentic — on top of the shared scaffolding in
`base.py`.

This page maps every exported template class to the
[`LLMProvider`][trulens.feedback.LLMProvider] method(s) that use it and the
use case it serves, so you can go from "I want to measure X" to the right
provider method quickly.

!!! tip "How to read this page"

    Most metrics expose two provider methods: a plain scorer (e.g.
    `relevance`) and a chain-of-thought variant that also returns reasons
    (e.g. `relevance_with_cot_reasons`). The chain-of-thought variants are
    generally recommended because the returned reasons make evaluations
    auditable.

## RAG metrics

Defined in `rag.py`. These power the
[RAG Triad](../../../getting_started/core_concepts/rag_triad.md) and related
retrieval-quality evaluations.

| Template Class | Provider Method(s) | Use Case |
|---|---|---|
| `Groundedness` | `groundedness_measure_with_cot_reasons`, `groundedness_measure_with_cot_reasons_consider_answerability` | Measure how well the response is supported by the retrieved context. |
| `ContextRelevance` | `context_relevance`, `context_relevance_with_cot_reasons` | Score whether each retrieved context chunk is relevant to the question. |
| `PromptResponseRelevance` | `relevance`, `relevance_with_cot_reasons` | Score whether the response is relevant to the prompt. |
| `Answerability` | `groundedness_measure_with_cot_reasons_consider_answerability` | Classify whether a question is answerable from the given source. |
| `Abstention` | `groundedness_measure_with_cot_reasons_consider_answerability` | Detect when a statement is an abstention ("I don't know"). |
| `Comprehensiveness` | `comprehensiveness_with_cot_reasons` | Measure how completely a summary covers the source's key points. |
| `Relevance` | (base class for `ContextRelevance` / `PromptResponseRelevance`) | Shared scaffolding for relevance-style metrics. |
| `GroundTruth` | (scaffolding) | Base template for ground-truth comparison metrics. |
| `Trivial` | (helper) | Identify trivial / low-information statements during grounding. |

## Quality metrics

Defined in `quality.py`. General-purpose text-quality evaluations.

| Template Class | Provider Method(s) | Use Case |
|---|---|---|
| `Conciseness` | `conciseness`, `conciseness_with_cot_reasons` | Score how concise the response is. |
| `Correctness` | `correctness`, `correctness_with_cot_reasons`, `model_agreement` | Score the factual correctness of the response. |
| `Coherence` | `coherence`, `coherence_with_cot_reasons` | Score how coherent and well-structured the response is. |
| `Sentiment` | `sentiment`, `sentiment_with_cot_reasons` | Score the positive sentiment of the response. |
| `Helpfulness` | `helpfulness`, `helpfulness_with_cot_reasons` | Score how helpful the response is. |
| `Controversiality` | `controversiality`, `controversiality_with_cot_reasons` | Score how controversial the response is. |

## Safety metrics

Defined in `safety.py`. Moderation and harm-detection evaluations.

| Template Class | Provider Method(s) | Use Case |
|---|---|---|
| `Stereotypes` | `stereotypes`, `stereotypes_with_cot_reasons` | Detect gender / race stereotyping introduced in the response. |
| `Harmfulness` | `harmfulness`, `harmfulness_with_cot_reasons` | Detect harmful content in the response. |
| `Maliciousness` | `maliciousness`, `maliciousness_with_cot_reasons` | Detect malicious intent in the response. |
| `Criminality` | `criminality`, `criminality_with_cot_reasons` | Detect content that promotes criminal activity. |
| `Insensitivity` | `insensitivity`, `insensitivity_with_cot_reasons` | Detect racially / culturally insensitive content. |
| `Misogyny` | `misogyny`, `misogyny_with_cot_reasons` | Detect misogynistic content. |
| `Toxicity` | (moderation category) | Moderation-style toxicity template. |
| `Maliciousness` | `maliciousness`, `maliciousness_with_cot_reasons` | Detect malicious content. |
| `HateThreatening` | (moderation category) | Moderation template for threatening hate speech. |
| `SelfHarm` | (moderation category) | Moderation template for self-harm content. |
| `Sexual` | (moderation category) | Moderation template for sexual content. |
| `SexualMinors` | (moderation category) | Moderation template for sexual content involving minors. |
| `Violence` | (moderation category) | Moderation template for violent content. |
| `GraphicViolence` | (moderation category) | Moderation template for graphic violence. |

!!! note

    Classes marked *(moderation category)* are templates for moderation-style
    categories and are consumed by moderation-endpoint providers rather than
    exposed as dedicated `LLMProvider` chain-of-thought methods.

## Agentic metrics

Defined in `agent.py`. Evaluations for agentic systems — planning, tool use,
and execution.

| Template Class | Provider Method(s) | Use Case |
|---|---|---|
| `LogicalConsistency` | `logical_consistency_with_cot_reasons` | Evaluate the logical consistency of the agent's reasoning. |
| `ExecutionEfficiency` | `execution_efficiency_with_cot_reasons` | Evaluate how efficiently the agent reached its goal. |
| `PlanAdherence` | `plan_adherence_with_cot_reasons` | Evaluate how well execution adhered to the plan. |
| `PlanQuality` | `plan_quality_with_cot_reasons` | Evaluate the quality of the agent's plan. |
| `ToolSelection` | `tool_selection_with_cot_reasons` | Evaluate the agent's choice of tools. |
| `ToolCalling` | `tool_calling_with_cot_reasons` | Evaluate the agent's tool invocation quality. |
| `ToolQuality` | `tool_quality_with_cot_reasons` | Evaluate the tool / system-side quality and reliability. |

## Scaffolding & shared building blocks

Defined in `base.py`. These are not user-facing metrics; they are the base
classes, enums, and prompt constants that the domain templates build on. The
most relevant ones for advanced users writing custom templates:

| Symbol | Role |
|---|---|
| `FeedbackTemplate`, `WithPrompt` | Base classes for all feedback templates. |
| `NaturalLanguage`, `Syntax`, `Semantics`, `LanguageMatch` | Intermediate base classes by evaluation family. |
| `Criteria`, `supported_criteria` | Named criteria and the criteria lookup table. |
| `CriteriaOutputSpaceMixin` | Mixin giving a template a configurable `criteria` + `output_space`. |
| `OutputSpace` | Enum of valid score output spaces (e.g. 0–3, 0–10, binary). |
| `EvalSchema`, `FewShotExample`, `FewShotExamples` | Few-shot example scaffolding. |
| `COT_REASONS_TEMPLATE`, `LIKERT_0_3_PROMPT`, `BINARY_0_1_PROMPT`, `LIKERT_0_10_PROMPT` | Shared prompt fragments reused across templates. |

The full list of exported symbols is available in
`trulens.feedback.templates.__all__`.
