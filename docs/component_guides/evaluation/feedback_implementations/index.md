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

`generate_score` and `generate_score_and_reasons` raise `trulens.feedback.generated.ParseError` whenever no score can be read from the judge's reply, and so do the `_with_cot_reasons` methods built on them. That covers free text without any number, a chain-of-thought reply with criteria and supporting evidence but no score line, and a JSON reply whose `score` is not a number. A score outside the configured scale raises `ParseError` too. When TruLens computes feedback on recorded app calls, the exception is recorded as that evaluation's error; `Jury` drops a juror that raises and still scores the remaining votes.

Some metrics record a failure as a score instead of raising: [`UNPARSABLE_SCORE`][trulens.core.utils.constants.UNPARSABLE_SCORE] (`-1.0`). The groundedness methods give it to a statement whose judge reply could not be parsed and average only the graded statements, returning it when none was graded; the role adherence metric returns it when it cannot read the judge's verdicts; and results stored by earlier versions may carry it. Scores are otherwise normalized to the 0 to 1 range, so a negative score is never a verdict. _TruLens_ skips it in its own aggregates: `Jury` drops that juror's vote, `BatchEvaluator` leaves it out of averages, and guardrails refuse to act on it.

If you call a `_with_cot_reasons` method directly and compare the score against a threshold, catch `ParseError` when calling most feedback methods, and check [`is_unparsable_score`][trulens.core.utils.constants.is_unparsable_score] on scores from groundedness and role adherence, which record failures as scores rather than raising. Both `UNPARSABLE_SCORE` and `is_unparsable_score` are also importable from `trulens.feedback`:

!!! example

    ```python
    from trulens.feedback import is_unparsable_score
    from trulens.feedback.generated import ParseError

    try:
        score, reasons = provider.coherence_with_cot_reasons(text)
    except ParseError as e:
        raise RuntimeError("judge reply had no score") from e

    score, reasons = provider.groundedness_measure_with_cot_reasons(
        source, statement
    )
    if is_unparsable_score(score):
        raise RuntimeError(f"no statement could be graded: {reasons}")
    ```

## Classification-based Providers

Some feedback functions rely on classification models, typically tailor-made for evaluation tasks, unlike LLM models.

This implementation consists of:

1. A call to a specific classification model useful for accomplishing a given evaluation task.
2. Custom logic to perform data preprocessing tasks before the classification model is called for evaluation.
3. Additional logic to perform postprocessing tasks using the classification model output.
