# Evaluation using Feedback Functions

## Why do you need feedback functions?

Measuring the performance of LLM apps is a critical step in the path from development to production. You would not move a traditional ML system to production without first gaining confidence by measuring its accuracy on a representative test set.

However unlike traditional machine learning, ground truth is sparse and often entirely unavailable.

Without ground truth on which to compute metrics on our LLM apps, feedback functions can be used to compute metrics for LLM applications.

## What is a feedback function?

Feedback functions, analogous to [labeling functions](https://arxiv.org/abs/2101.07138), provide a programmatic method for generating evaluations on an application run. In our view, this method of evaluations is far more useful than general benchmarks because they
measure the performance of **your app, on your data, for your users**.

!!! info "Important Concept"

    TruLens constructs feedback functions by combining more general models, known as the [**_feedback provider_**][trulens_eval.feedback.provider.base.Provider], and [**_feedback implementation_**](../feedback_implementations/index.md) made up of carefully constructed prompts and custom logic tailored to perform a particular evaluation task.

This construction is **composable and extensible**.

**Composable** meaning that the user can choose to combine any feedback provider with any feedback implementation.

**Extensible** meaning that the user can extend a feedback provider with custom feedback implementations of the user's choosing.

!!! example

    In a high stakes domain requiring evaluating long chunks of context, the user may choose to use a more expensive SOTA model.

    In lower stakes, higher volume scenarios, the user may choose to use a smaller, cheaper model as the provider.

    In either case, any feedback provider can be combined with a _TruLens_ feedback implementation to ultimately compose the feedback function.
