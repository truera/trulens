# ✅ Standards

Enumerations of standards for code and its documentation to be maintained in
`trulens`. Ongoing work aims at adapting these standards to existing code.

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

### Format

- See `pyproject.toml` section `[tool.ruff]`.

### Imports

- See `pyproject.toml` section `[tool.ruff.lint.isort]` on tooling to organize
  import statements.

- Generally import modules only as per
  <https://google.github.io/styleguide/pyguide.html#22-imports>. That us:

    ```python
    from trulens.schema.record import Record # don't do this
    from trulens.schema import record as record_schema # do this instead
    ```

    This prevents the `record` module from being loaded until something inside it
    is needed. If your uses of `record_schema.Record` are inside functions, this
    loading can be delayed as far as the execution of that function.

- Import and rename modules:

    ```python
    from trulens.schema import record # don't do this
    from trulens.schema import record as record_schema # do this
    ```

    This is especially important for module names which might cause name
    collisions with other things such as variables named `record`.

- Keep module renames consistent using the following patterns:

    ```python
    # schema
    from trulens.schema import X as X_schema

    # utils
    from trulens.utils import X as X_utils # if X was plural, make X singular in rename

    # providers
    from trulens.providers.X import provider as X_provider
    from trulens.providers.X import endpoint as X_endpoint

    # apps
    from trulens.apps.X import Y as Y_app

    # connectors
    from trulens.connector import X as X_connector

    # core modules
    from trulens.core import X as core_X

    # core.feedback modules
    from trulens.core.feedback import X as core_X

    # core.database modules
    from trulens.core.database import base as core_db
    from trulens.core.database import connector as core_connector
    from trulens.core.database import X as X_db

    # dashboard modules
    from trulens.dashboard.X import Y as dashboard_Y

    # if X is inside some category of module Y:
    from trulens...Y import X as X_Y
    # otherwise if X is not in some category of modules:
    from trulens... import X as mod_X

    # Some modules do not need renaming:
    from trulens.feedback import llm_provider
    ```

- If an imported module is only used in type annotations, import it inside a
  `TYPE_CHECKING` block:

    ```python
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
      from trulens.schema import record as record_schema
    ```

- Do not create exportable aliases (an alias that is listed in `__all__` and
  refers to an element from some other module). Don't import aliases. Type
  aliases, even exportable ones are ok:

    ```python
    Thunk[T] = Callable[[], T] # OK
    AppID = types_schema.AppID # not OK
    ```

#### Circular imports

Circular imports may become an issue (error when executing your/`trulens` code,
indicated by phrase "likely due to circular imports"). The Import guideline
above may help alleviate the problem. A few more things can help:

- Use annotations feature flag:

    ```python
    from __future__ import annotations
    ```

    However, if your module contains `pydantic` models, you may need to run
    `model_rebuild`:

    ```python
    from __future__ import annotations

    ...

    class SomeModel(pydantic.BaseModel):

      some_attribute: some_module.SomeType

    ...

    SomeModel.model_rebuild()
    ```

    If you have multiple mutually referential models, you may need to rebuild only
    after all are defined.


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
    attribute_name: Description.

    attribute_name: Description.
"""
````

For pydantic classes, provide the attribute description as a long string right
after the attribute definition:

```python
class SomeModel(pydantic.BaseModel)
  """Class summary

  Class details.
  """

  attribute: Type = defaultvalue # or pydantic.Field(...)
  """Summary as first sentence.

  Details as the rest.
  """

  cls_attribute: typing.ClassVar[Type] = defaultvalue # or pydantic.Field(...)
  """Summary as first sentence.

  Details as the rest.
  """

  _private_attribute: Type = pydantic.PrivateAttr(...)
  """Summary as first sentence.

  Details as the rest.
  """

```

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

    Additional return discussion. Use list above to point out return components if there are multiple relevant components.

Raises:
    ExceptionType: Description.
"""
````

Note that the types are automatically filled in by docs generator from the
function signature.

## Typescript

No standards are currently recommended.

## Markdown

- Always indicate code type in code blocks as in python in

    ````markdown
    ```python
    # some python here
    ```
    ````

  Relevant types are `python`, `typescript`, `json`, `shell`, `markdown`.
  Examples below can serve as a test of the markdown renderer you are viewing
  these instructions with.

  - Python
    ```python
    a = 42
    ```

  - Typescript
    ```typescript
    var a = 42;
    ```

  - JSON
    ```json
    {'a': [1,2,3]}
    ```

  - Shell
    ```shell
    > make test-api
    > pip install trulens
    ```

  - Markdown
    ```markdown
    # Section heading
    content
    ```

- Use `markdownlint` to suggest formatting.

- Use 80 columns if possible.

## Jupyter notebooks

Do not include output. The pre-commit hooks should automatically clear all
notebook outputs.

## Tests

### Unit tests

See `tests/unit`.

### Static tests

See `tests/unit/static`.

Static tests run on multiple versions of python: `3.8`, `3.9`, `3.10`, `3.11`,
and being a subset of unit tests, are also run on latest supported python,
`3.12` . Some tests that require all optional packages to be installed run only
on `3.11` as the latter python version does not support some of those optional
packages.

### Test pipelines

Defined in `.azure_pipelines/ci-eval{-pr,}.yaml`.
