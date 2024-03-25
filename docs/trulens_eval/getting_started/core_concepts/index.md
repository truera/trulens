# ⭐ Core Concepts

- ☔ [Feedback Functions](feedback_functions.md).

- ⟁ [Rag Triad](rag_triad.md).

- 🏆 [Honest, Harmless, Helpful Evals](honest_harmless_helpful_evals.md).

## Glossary

- `Agent`.

- `Application` or `App`. An "application" that is tracked by _TruLens-Eval_.
  Abstract definition of this tracking corresponds to
  [App][trulens_eval.app.App].

- `Completion`, `Generation`.

- `Component`.

- `Embedding`.

- `Eval`, `Evals`, `Evaluation`.

- `Feedback`.

- `Feedback Function`.

- `Generation`. See `Completion`.

- `Human Feedback`.

- `Prompt`.

- `Provider`.

- `RAG`, `Retrieval Augmented Generation`.

- `Record`. A "record" of the execution of a single execution of an app. Single
  execution means invocation of some top-level app method. Corresponds to
  [Record][trulens_eval.schema.Record].
  
    !!! note
        This will be renamed to `Trace` in the future.
  
- `Span`. Some unit of work logged as part of a record. Corresponds to current
  [RecordAppCallMethod][trulens_eval.schema.RecordAppCall].

- `Tool`.

- `Trace`. See `Record`.
