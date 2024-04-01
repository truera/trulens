As a reminder, a typical feedback definition looks like this:

```python
f_lang_match = Feedback(hugs.language_match)
    .on_input_output()
```

`on_input_output` is one of many available shortcuts to simplify the selection
of components for evaluation.

The selector, `on_input_output`, specifies how the `language_match` arguments
are to be determined from an app record or app definition. The general form of
this specification is done using `on` but several shorthands are provided.
`on_input_output` states that the first two argument to `language_match`
(`text1` and `text2`) are to be the main app input and the main output,
respectively.

Several utility methods starting with `.on` provide shorthands:

- `on_input(arg) == on_prompt(arg: Optional[str])` -- both specify that the next
  unspecified argument or `arg` should be the main app input.

- `on_output(arg) == on_response(arg: Optional[str])` -- specify that the next
  argument or `arg` should be the main app output.

- `on_input_output() == on_input().on_output()` -- specifies that the first two
  arguments of implementation should be the main app input and main app output,
  respectively.

- `on_default()` -- depending on signature of implementation uses either
  `on_output()` if it has a single argument, or `on_input_output` if it has two
  arguments.

Some wrappers include additional shorthands:

### LlamaIndex specific selectors

- `TruLlama.select_source_nodes()` -- outputs the selector of the source
  documents part of the engine output.

  Usage:

  ```python
  from trulens_eval import TruLlama
  source_nodes = TruLlama.select_source_nodes(query_engine)
  ```

- `TruLlama.select_context()` -- outputs the selector of the context part of the
  engine output.

  Usage:

  ```python
  from trulens_eval import TruLlama
  context = TruLlama.select_context(query_engine)
  ```

### _LangChain_ specific selectors

- `TruChain.select_context()` -- outputs the selector of the context part of the
  engine output.

  Usage:

  ```python
  from trulens_eval import TruChain
  context = TruChain.select_context(retriever_chain)
  ```

### _LlamaIndex_ and _LangChain_ specific selectors

- `App.select_context()` -- outputs the selector of the context part of the
  engine output. Can be used for both _LlamaIndex_ and _LangChain_ apps.

  Usage:

  ```python
  from trulens_eval.app import App
  context = App.select_context(rag_app)
  ```
