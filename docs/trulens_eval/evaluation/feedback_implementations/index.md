# Feedback Implementations

TruLens constructs feedback functions by a [**_feedback provider_**][trulens_eval.feedback.provider.base.Provider], and [**_feedback implementation_**](../feedback_implementations/index.md).

This page documents the feedback implementations available in _TruLens_.

Feedback functions are implemented in instances of the [Provider][trulens_eval.feedback.provider.base.Provider] class. They are made up of carefully constructed prompts and custom logic tailored to perform a particular evaluation task.

## Generation-based feedback implementations

The implementation of generation-based feedback functions can consist of:

1. Instructions to a generative model (LLM) on how to perform a particular evaluation task. These instructions are sent to the LLM as a system message, and often consist of a rubric.
2. A template that passes the arguments of the feedback function to the LLM. This template containing the arguments of the feedback function is sent to the LLM as a user message.
3. A method for parsing, validating, and normalizing the output of the LLM, accomplished by [`generate_score`][trulens_eval.feedback.provider.base.LLMProvider.generate_score].
4. Custom Logic to perform data preprocessing tasks before the LLM is called for evaluation.
5. Additional logic to perform postprocessing tasks using the LLM output.

_TruLens_ can also provide reasons using [chain-of-thought methodology](https://arxiv.org/abs/2201.11903). Such implementations are denoted by method names ending in `_with_cot_reasons`. These implementations illicit the LLM to provide reasons for its score, accomplished by [`generate_score_and_reasons`][trulens_eval.feedback.provider.base.LLMProvider.generate_score_and_reasons].

## Classification-based Providers

Some feedback functions rely on classification models, typically tailor made for task, unlike LLM models.

This implementation consists of:

1. A call to a specific classification model useful for accomplishing a given evaluation task.
2. Custom Logic to perform data preprocessing tasks before the classification model is called for evaluation.
3. Additional logic to perform postprocessing tasks using the classification model output.
