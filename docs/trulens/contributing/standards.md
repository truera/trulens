# ✅ Standards

Enumerations of standards for code and its documentation to be maintained in
_TruLens_. Ongoing work aims at adapting these standards to existing code.

## Proper Names

In natural language text, style/format proper names using italics if available.
In Markdown, this can be done with a single underscore character on both sides
of the term. In unstyled text, use the capitalization as below. This does not
apply when referring to things like package names, classes, methods.

- _TruLens_

- _LangChain_

- _LlamaIndex_

- _NeMo Guardrails_

- _OpenAI_

- _Bedrock_

- _LiteLLM_

- _Pinecone_

- _HuggingFace_

## Python

### Formatting

- Use `ruff` to format code with configuration:

  ```toml
  [tool.ruff]
  line-length = 80
  ```

  Additionally configure `ruff` with `lint` and `isort` as described later in
  this document. A full `ruff` configuration is found in the root
  `pyproject.toml`.

- Use `ruff.lint` for various code issues:

    ```toml
    [tool.ruff.lint.pydocstyle]
    convention = "google"
    ```

### Imports

- Use `ruff.lint.isort` to organize import statements:

  ```toml
    [tool.ruff.lint.isort]
    force-single-line = true
    force-sort-within-sections = true
    single-line-exclusions = [
      "typing",
    ]
    known-first-party = [
      "src",
    ]
  ```

- Generally import modules only as per
  <https://google.github.io/styleguide/pyguide.html#22-imports> with some
  exceptions:

  - Very standard names like types from python or widely used packages. Also
    names meant to stand in for them.
  - Other exceptions in the google style guide above.

- Use full paths when importing internally
  <https://google.github.io/styleguide/pyguide.html#23-packages>. Aliases still
  ok for external users.

### Docstrings

- Docstring placement and low-level issues <https://peps.python.org/pep-0257/>.

- Content is formatted according to
  <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>.

#### Example: Modules

````markdown
"""Summary line.

More details if necessary.

Design:

Discussion of design decisions made by module if appropriate.

Examples:

```python
# example if needed
```

Deprecated:
    Deprecation points.
"""
````

#### Example: Classes

````markdown
"""Summary line.

More details if necessary.

Example:

  ```python
  # example if needed
  ```

Attrs:
    attribute_name: Description.

    attribute_name: Description.
"""
````

#### Example: Functions/Methods

````markdown
"""Summary line.

More details if necessary.

Example:

  ```python
  # example if needed
  ```

Args:
    argument_name: Description. Some long description of argument may wrap over to the next line and needs to
        be indented there.

    argument_name: Description.

Returns:

    return_type: Description.

    return_type: More descriptions if there is more than one return value.

    Additional return discussion. Use list above to point out return components if there are multiple relevant components.

Raises:

    ExceptionType: Description.
"""
````

Note that the types are automatically filled in by docs generator from the
function signature.

## Markdown

- Always indicate code type in code blocks as in python in

    ````markdown
    ```python
    # some python here
    ```
    ````

- Use `markdownlint` to suggest formatting.

- Use 80 columns if possible.

## Jupyter notebooks

Do not include output unless core goal of given notebook.

## Tests

### Unit tests

See `tests/unit`.

### Static tests

See `tests/unit/static`.

Static tests run on multiple versions of python: `3.8`, `3.9`, `3.10`, `3.11`, and being a
subset of unit tests, are also run on latest supported python, `3.12` .

### Test pipelines

Defined in `.azure_pipelines/ci-eval{-pr,}.yaml`.
