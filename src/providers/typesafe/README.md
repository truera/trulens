# TruLens TypeSafe provider

This package integrates [TypeSafe Jev](https://docs.typesafe.ai/introduction),
a non-generative System One decision model, with TruLens feedback functions.
Jev evaluates typed questions against shared state and returns constrained
values and probability distributions.

```bash
pip install trulens-providers-typesafe
export TYPESAFE_API_KEY="your-api-key"
```

```python
from trulens.providers.typesafe import TypeSafe

provider = TypeSafe()  # Pinned to jev-1.13.0 by default.
score = provider.context_relevance(
    question="Where is the Eiffel Tower?",
    context="The Eiffel Tower is in Paris.",
)
```

Use `evaluate` to batch any mix of official SDK `Noul`, `Choice`, and `Score`
questions that share state. It returns a `TypeSafeDecision` for each question,
including the raw distribution, model, primitive, question-template hash,
ordered criteria, state hash, and token usage.

Jev does not generate explanations. Methods ending in
`_with_cot_reasons` therefore raise `NotImplementedError` instead of silently
falling back to another model.
