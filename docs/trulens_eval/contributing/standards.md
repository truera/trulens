# ✅ Standards

Enumerations of standards for code and its documentation to be maintained in
`trulens_eval`. Ongoing work aims at adapting these standards to existing code.

## Proper Names

Styling/formatting of proper names.

- "TruLens"

- "LangChain"

- "LlamaIndex"

- "NeMo Guardrails", "Guardrails" for short, "rails" for shorter.

## Python

### Format

- Use `pylint` for various code issues.

- Use `yapf` to format code with configuration:

    ```toml
    [style]
    based_on_style = google
    DEDENT_CLOSING_BRACKETS=true
    SPLIT_BEFORE_FIRST_ARGUMENT=true
    SPLIT_COMPLEX_COMPREHENSION=true
    COLUMN_LIMIT=80
    ```

### Imports

- Use `isort` to organize import statements.

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

Examples:

```python
# example if needed
```

Attrs:
    attribute_name (attribute_type): Description.

    attribute_name (attribute_type): Description.
"""
````

#### Example: Functions/Methods

````markdown
"""Summary line.

More details if necessary.

Examples:

```python
# example if needed
```

Args:
    argument_name: Description. Some long description of argument may wrap over to the next line and needs to
        be indented there.

    argument_name: Description.

Returns:

    return_type: Description.

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

Static tests run on multiple versions of python: 3.8, 3.9, 3.10, and being a
subset of unit tests, are also run on latest supported python, 3.11.

### Test pipelines

Defined in `.azure_pipelines/ci-eval{-pr,}.yaml`.
