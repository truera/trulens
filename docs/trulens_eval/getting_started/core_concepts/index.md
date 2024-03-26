# ⭐ Core Concepts

- ☔ [Feedback Functions](feedback_functions.md).

- ⟁ [Rag Triad](rag_triad.md).

- 🏆 [Honest, Harmless, Helpful Evals](honest_harmless_helpful_evals.md).

## Glossary

General and 🦑_TruLens-Eval_-specific concepts.

- `Agent`. A `Component` of an `Application` that performs some related set of
  tasks potentially interfacing with some external services or APIs.

- `Application` or `App`. An "application" that is tracked by 🦑_TruLens-Eval_.
  Abstract definition of this tracking corresponds to
  [App][trulens_eval.app.App]. We offer special support for _LangChain_ via
  [TruChain][trulens_eval.tru_chain.TruChain], _LlamaIndex_ via
  [TruLlama][trulens_eval.tru_llama.TruLlama], and _NeMo Guardrails_ via
  [TruRails][trulens_eval.tru_rails.TruRails] `Applications` as well as custom
  apps via [TruBasicApp][trulens_eval.tru_basic_app.TruBasicApp] or
  [TruCustomApp][trulens_eval.tru_custom_app.TruCustomApp], and apps that
  already come with `Trace`s via
  [TruVirtual][trulens_eval.tru_virtual.TruVirtual].

- `Chain`. A _LangChain_ `App`.

- `Completion`, `Generation`. The process or result of LLM responding to some
  `Prompt`.

- `Component`. Part of an `Application`.

- `Embedding`. A real vector representation of some piece of text. Can be used
  to find related pieces of text in a `Retrieval`.

- `Eval`, `Evals`, `Evaluation`. Process or result of method that scores the
  outputs or aspects of a `Trace`. In 🦑_TruLens-Eval_, our scores are real
  numbers between 0 and 1.

- `Feedback`. See `Evaluation`.

- `Feedback Function`. A method that implements an `Evaluation`. This
  corresponds to [Feedback][trulens_eval.feedback.feedback.Feedback].

- `Generation`. See `Completion`.

- `Human Feedback`. A feedback that is provided by a human, e.g. a thumbs
  up/down in response to a `Completion`.

- `LLM`, `Large Language Model`. The `Component` of an `Application` that
  performs `Completion`.

- `Prompt`. The text that an `LLM` completes during `Completion`. In chat
  applications, the user's message.

- `Provider`. A system that _provides_ the ability to execute models, either
  `LLM`s or classification models. In 🦑_TruLens-Eval_, `Feedback Functions`
  make use of `Providers` to invoke models for `Evaluation`.

- `RAG`, `Retrieval Augmented Generation`. A common organization of
  `Applications` that combine a `Retrieval` with an `LLM` to produce
  `Completions` that incorporate information that an `LLM` alone may not be
  aware of.

- `RAG Triad` (🦑_TruLens-Eval_-specific concept). A combination of three
  `Feedback Functions` meant to `Evaluate` `Retrieval` steps in `Applications`.

- `Record`. A "record" of the execution of a single execution of an app. Single
  execution means invocation of some top-level app method. Corresponds to
  [Record][trulens_eval.schema.Record].
  
    !!! note
        This will be renamed to `Trace` in the future.

- `Retrieval`. The process or result of looking up pieces of context relevant to
  some query. Typically this is done using an `Embedding` reprqesentations of
  queries and contexts.

- `Selector` (🦑_TruLens-Eval_-specific concept). A specification of the source
  of data from a `Trace` to use as inputs to a `Feedback Function`. This
  corresponds to [Lens][trulens_eval.utils.serial.Lens] and utilities
  [Select][trulens_eval.schema.Select].

- `Span`. Some unit of work logged as part of a record. Corresponds to current
  🦑[RecordAppCallMethod][trulens_eval.schema.RecordAppCall].

- `Tool`. See `Agent`.

- `Trace`. See `Record`.
