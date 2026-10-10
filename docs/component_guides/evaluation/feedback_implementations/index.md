# Feedback Implementations

TruLens constructs feedback functions by a [**_feedback provider_**][trulens.core.feedback.Provider], and **_feedback implementation_**.

This page documents the feedback implementations available in _TruLens_.

Feedback functions are implemented in instances of the [Provider][trulens.core.feedback.Provider] class. They are made up of carefully constructed prompts and custom logic tailored to perform a particular evaluation task.

## Generation-based feedback implementations

The implementation of generation-based feedback functions can consist of:

1. Instructions to a generative model (LLM) on how to perform a particular evaluation task. These instructions are sent to the LLM as a system message, and often consist of a rubric.
2. A template that passes the arguments of the feedback function to the LLM. This template containing the arguments of the feedback function is sent to the LLM as a user message.
3. A method for parsing, validating, and normalizing the output of the LLM, accomplished by [`generate_score`][trulens.feedback.LLMProvider.generate_score].
4. Custom logic to perform data preprocessing tasks before the LLM is called for evaluation.
5. Additional logic to perform postprocessing tasks using the LLM output.

_TruLens_ can also provide reasons using [chain-of-thought methodology](https://arxiv.org/abs/2201.11903). Such implementations are denoted by method names ending in `_with_cot_reasons`. These implementations illicit the LLM to provide reasons for its score, accomplished by [`generate_score_and_reasons`][trulens.feedback.LLMProvider.generate_score_and_reasons].

### When the judge reply cannot be parsed

`generate_score` and `generate_score_and_reasons` return [`UNPARSABLE_SCORE`][trulens.core.utils.constants.UNPARSABLE_SCORE] (`-1.0`) when no score can be read out of the judge's reply, on both the JSON and the plain text path. Scores are otherwise normalized to the 0 to 1 range, so a negative score is never a verdict. TruLens's own aggregates skip it: `Jury` drops that juror's vote, `BatchEvaluator` leaves it out of averages, the groundedness implementation averages only the statements that were graded, and guardrails refuse to act on it.

If you call a `_with_cot_reasons` method directly and compare the score against a threshold, check [`is_unparsable_score`][trulens.core.utils.constants.is_unparsable_score] first. Both names are also importable from `trulens.feedback`:

!!! example

    ```python
    from trulens.feedback import UNPARSABLE_SCORE, is_unparsable_score

    score, reasons = provider.coherence_with_cot_reasons(text)
    if is_unparsable_score(score):
        raise RuntimeError(f"judge produced no score: {reasons}")
    ```

## Classification-based Providers

Some feedback functions rely on classification models, typically tailor-made for evaluation tasks, unlike LLM models.

This implementation consists of:

1. A call to a specific classification model useful for accomplishing a given evaluation task.
2. Custom logic to perform data preprocessing tasks before the classification model is called for evaluation.
3. Additional logic to perform postprocessing tasks using the classification model output.
