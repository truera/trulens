# Anatomy of Feedback Functions

The `Feedback` class contains the starting point for feedback function
specification and evaluation. A typical use-case looks like this:

```python
from trulens_eval import OpenAI

openai = OpenAI(model_engine="gpt-3.5-turbo")

f_relevance = Feedback(openai.relevance).on_input_output()
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
  def relevance(self, prompt: str, response: str) -> float:
  ```

  That is, `relevance` is a plain python method that accepts the prompt and
  response, both strings, and produces a float (assumed to be between 0.0 and
  1.0).

- **Feedback constructor** -- The line `Feedback(openai.relevance)` constructs a
  Feedback object with a feedback implementation.

- **Argument specification** -- The next line, `on_input_output`, specifies how
  the `language_match` arguments are to be determined from an app record or app
  definition. The general form of this specification is done using `on` but
  several shorthands are provided. For example, `on_input_output` states that the first two
  argument to `relevance` (`prompt` and `response`) are to be the main app input
  and the main output, respectively.
