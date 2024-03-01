# Feedback Functions

Measuring the performance of LLM apps is a critical step in the path from
development to production. You would not move a traditional ML system to
production without first gaining confidence by measuring its accuracy on a test
set.

However unlike traditional machine learning, user feedback or any "ground truth"
is largely unavailable. Without ground truth on which to compute metrics on our
LLM apps, we can turn to feedback functions as a way to compute metrics for LLM
apps.

Feedback functions, analogous to labeling functions, provide a programmatic
method for generating evaluations on an application run. In our view, this
method of evaluations is far more useful than general benchmarks because they
measures the performance of **your app, on your data, for your users**.

Last, feedback functions are flexible. They can be implemented with any model on
the back-end. This includes rule-based systems, smaller models tailored to a
particular task, or carefully prompted large language models.
