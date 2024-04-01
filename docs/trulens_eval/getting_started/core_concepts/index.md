# ⭐ Core Concepts

- ☔ [Feedback Functions](feedback_functions.md).

- ⟁ [Rag Triad](rag_triad.md).

- 🏆 [Honest, Harmless, Helpful Evals](honest_harmless_helpful_evals.md).

## Glossary

General and 🦑_TruLens-Eval_-specific concepts.

- `Agent`. A `Component` of an `Application` or the entirety of an application
  that providers a natural language interface to some set of capabilities
  typically incorporating `Tools` to invoke or query local or remote services,
  while maintaining its state via `Memory`. The user of an agent may be a human, a
  tool, or another agent. See also `Multi Agent System`.

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

- `Chain of Thought`. The use of an `Agent` to deconstruct its tasks and to
  structure, analyze, and refine its `Completions`.

- `Completion`, `Generation`. The process or result of LLM responding to some
  `Prompt`.

- `Component`. Part of an `Application` giving it some capability. Typical
  components include:

  - `Retriever`
  
  - `Memory`
  
  - `Tool`

  - `Prompt Template`

  - `LLM`

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

- `Instruction Prompt`, `System Prompt`. A part of a `Prompt` given to an `LLM`
  to complete that contains instructions describing the task that the
  `Completion` should solve. Sometimes such prompts include examples of correct
  or desirable completions (see `Shots`). A prompt that does not include examples
  is said to be `Zero Shot`.

- `LLM`, `Large Language Model`. The `Component` of an `Application` that
  performs `Completion`.

- `Memory`. The state maintained by an `Application` or an `Agent` indicating
  anything relevant to continuing, refining, or guiding it towards its
  goals. `Memory` is provided as `Context` in `Prompts` and is updated when new
  relevant context is processed, be it a user prompt or the results of the
  invocation of some `Tool`. As `Memory` is included in `Prompts`, it can be a
  natural language description of the state of the app/agent. To limit to size
  if memory, `Summarization` is often used.

- `Multi-Agent System`. The use of multiple `Agents` incentivized to interact
  with each other to implement some capability. While the term predates `LLMs`,
  the convenience of the common natural language interface makes the approach
  much easier to implement.

- `Prompt`. The text that an `LLM` completes during `Completion`. In chat
  applications. See also `Instruction Prompt`, `Prompt Template`.

- `Prompt Template`. A piece of text with placeholders to be filled in in order
  to build a `Prompt` for a given task. A `Prompt Template` will typically
  include the `Instruction Prompt` with placeholders for things like `Context`,
  `Memory`, or `Application` configuration parameters.

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

- `Retrieval`, `Retriever`. The process or result (or the `Component` that
  performs this) of looking up pieces of text relevant to a `Prompt` to provide
  as `Context` to an `LLM`. Typically this is done using an `Embedding`
  representations.

- `Selector` (🦑_TruLens-Eval_-specific concept). A specification of the source
  of data from a `Trace` to use as inputs to a `Feedback Function`. This
  corresponds to [Lens][trulens_eval.utils.serial.Lens] and utilities
  [Select][trulens_eval.schema.Select].

- `Shot`, `Zero Shot`, `Few Shot`, `<Quantity>-Shot`. The use of zero or more
  examples in an `Instruction Prompt` to help an `LLM` generate desirable
  `Completions`. `Zero Shot` describes prompts that do not have any examples and
  only offer a natural language description of the task, while `<Quantity>-Shot`
  indicate some `<Quantity>` of examples are provided.

- `Span`. Some unit of work logged as part of a record. Corresponds to current
  🦑[RecordAppCallMethod][trulens_eval.schema.RecordAppCall].

- `Summarization`. The task of condensing some natural language text into a
  smaller bit of natural language text that preserves the most important parts
  of the text. This can be targetted towards humans or otherwise. It can also be
  used to maintain consize `Memory` in an `LLM` `Application` or `Agent`.
  Summarization can be performed by an `LLM` using a specific `Instruction Prompt`.

- `Tool`. A piece of functionality that can be invoked by an `Application` or
  `Agent`. This commonly includes interfaces to services such as search (generic
  search via google or more specific like IMDB for movies). Tools may also
  perform actions such as submitting comments to github issues. A `Tool` may
  also encapsulate an interface to an `Agent` for use as a component in a larger
  `Application`.

- `Trace`. See `Record`.
