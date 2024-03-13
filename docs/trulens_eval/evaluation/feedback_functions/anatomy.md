# Anatomy of Feedback Functions

The `Feedback` class contains the starting point for feedback function
specification and evaluation. A typical use-case looks like this:

```python
# Context relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name = "Context Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on(Select.RecordCalls.retrieve.rets)
    .aggregate(numpy.mean)
)
```

The components of this specifications are:

- **Feedback Providers** -- The provider is the back-end on which a given
  feedback function is run.' Multiple underlying models are available through
  each provider, such as GPT-4 or Llama-2. In many, but not all cases, the
  feedback implementation is shared across providers (such as with LLM-based
  evaluations).

- **Feedback implementations** -- `openai.relevance` is a feedback function
  implementation. Feedback implementations are simple callables that can be run
  on any arguments matching their signatures. In the example, the implementation
  has the following signature:

  ```python
  def context_relevance(self, prompt: str, context: str) -> float:
  ```

  That is, `context_relevance` is a plain python method that accepts the prompt and
  context, both strings, and produces a float (assumed to be between 0.0 and
  1.0).

- **Feedback constructor** -- The line `Feedback(openai.relevance)` constructs a
  Feedback object with a feedback implementation.

- **Argument specification** -- The next line, `on_input_output`, specifies how
  the `language_match` arguments are to be determined from an app record or app
  definition. The general form of this specification is done using `on` but
  several shorthands are provided. For example, `on_input_output` states that the first two
  argument to `relevance` (`prompt` and `response`) are to be the main app input
  and the main output, respectively.

- **Aggregation specification** -- The last line `aggregate(numpy.mean)` specifies
  how feedback outputs are to be aggregated. This only applies to cases where
  the argument specification names more than one value for an input. The second
  specification, for `statement` was of this type. The input to `aggregate` must
  be a method which can be imported globally. This requirement is further
  elaborated in the next section. This function is called on the `float` results
  of feedback function evaluations to produce a single float. The default is
  `numpy.mean`.
