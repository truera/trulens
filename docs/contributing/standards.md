# ✅ Standards

Standards for code and its documentation to be maintained in
`trulens`. Ongoing work aims at adapting these standards to existing code.

## AI-assisted contributions

Using an assistant to write _TruLens_ code is fine. The certification in the pull
request template is an assertion about the code, not about how it was produced: you
are the author, and you are answerable for it.

Before opening the pull request:

- Read every line you are submitting and be ready to say why it is written that
  way. If a reviewer asks and you cannot answer, it was not ready.
- Run it. `make format`, `make lint`, and the relevant `make test-*` target.
  Generated code often reads correctly and fails on import.
- Check the APIs exist. Assistants invent plausible _TruLens_ methods, attributes
  and keyword arguments. The
  [reference docs](https://www.trulens.org/reference/) are generated from the
  source, so they are the thing to check against.
- Check imports against [Imports](#imports) below. Generated code defaults to
  `from trulens.core.feedback.feedback import Feedback`; this project imports
  the module and renames it.
- Check the docstring format. Google style, as in
  [Docstrings](#docstrings) below.
- Check the tests assert something. A test where a mock returns what the mock was
  configured to return will pass and demonstrate nothing.
- Don't add a dependency to make generated code work. Optional dependencies have a
  structure; see
  [Optional Packages](https://www.trulens.org/contributing/optional/).
- Clear notebook outputs. The `nb-clean` pre-commit hook does this; run
  `pre-commit run --all-files` if you are unsure.

On scope:

- Prefer one reviewable change to one large one. Review is the scarce resource
  here, not authoring.
- Don't claim more issues than you are actively working on, and check whether
  someone else has claimed one before starting.
- Don't submit documentation or examples for behavior you have not run.

You are not required to disclose assistant use and reviewers will not ask. Mentioning
it is welcome, and pointing at the parts you are least sure of is more useful than a
blanket disclosure.

## Contributing your own integration

Integrations for a specific model, framework, vector store, or product are welcome,
and several of the packages here started that way. The guidance below is about where
that code sits, so that one integration does not take up more room in the project
than the others.

Keep it self-contained. Each integration is its own installable package, with its own
`pyproject.toml`, under `src/providers/`, `src/apps/`, or `src/connectors/`. That is
where the code, its optional dependencies, and its tests belong. See
[Optional Packages](https://www.trulens.org/contributing/optional/).

Leave the shared surfaces alone:

- Don't add a dependency on your integration to `trulens-core` or to any other
  package that does not need it.
- Don't change a core API to suit one integration. If core genuinely needs to
  change, make that case on its own in its own pull request, and it will be judged
  on whether it helps every integration.
- Don't add your product to the README, the quickstarts, or the top level of the
  documentation navigation as part of the same pull request, and don't reorder an
  existing list to move yours up it. Whether an integration is listed in the README
  is a separate decision from whether it is merged; ask rather than assume.
- Examples go in `examples/expositional/` and the cookbook, organized by topic.
  `examples/quickstart/` is for minimal demonstrations of core concepts with few
  dependencies, and is not the place to introduce a new integration.

Tests must run without your service. CI has no credentials for third-party APIs, so
tests need to pass with mocks or be marked so they can be skipped. The available
markers are in `pyproject.toml` under `[tool.pytest.ini_options]`; add one for your
integration if none fits.

Expect to keep it working. An integration is a standing commitment rather than a
one-off contribution: upstream SDKs change, and whoever added the integration is
usually the only person who can tell a real break from an intended change. An
integration nobody maintains will eventually be deprecated, following
[Release Policies](https://www.trulens.org/contributing/policies/). If you would
rather not carry that, a cookbook example is a good contribution on its own and
carries no such expectation.

## Proper Names

In natural language text, style/format proper names using italics if available.
In Markdown, this can be done with a single underscore character on both sides
of the term. In unstyled text, use the capitalization as below. This does not
apply when referring to things like package names, classes, or methods.

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
  <https://google.github.io/styleguide/pyguide.html#22-imports>. That is:

    ```python
    from trulens.schema.record import Record  # don't do this
    from trulens.schema import record as record_schema  # do this instead
    ```

    This approach prevents the `record` module from being loaded until something inside it
    is needed. If your uses of `record_schema.Record` are inside functions, this
    loading can be delayed as far as the execution of that function.

- Import and rename modules:

    ```python
    from trulens.schema import record  # don't do this
    from trulens.schema import record as record_schema  # do this
    ```

    This is especially important for module names, which might cause name
    collisions with other things such as variables named `record`.

- Keep module renames consistent using the following patterns (see `src/core/trulens/_mods.py` for the full list):

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
    Thunk[T] = Callable[[], T]  # OK
    AppID = types_schema.AppID  # not OK
    ```

#### Circular imports

Circular imports may become an issue (error when executing your/`trulens` code,
indicated by the phrase "likely due to circular imports"). The Import guideline
above may help alleviate the problem. A few more things can help:

- Use annotations feature flag:

    ```python
    from __future__ import annotations
    ```

    However, if your module contains `Pydantic` models, you may need to run
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

Additional details can be provided here if necessary.

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

Additional details can be provided here if necessary.

Examples:

```python
# example if needed
```

Attrs:
    attribute_name: Description.

    attribute_name: Description.
"""
````

For Pydantic classes, provide the attribute description as a long string right
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

The dashboard's React components are formatted with Prettier and linted with
ESLint. Configuration lives beside each component:

- `src/dashboard/react_components/record_viewer/.eslintrc.cjs` and `.prettierrc`
- `src/dashboard/react_components/record_viewer_otel/.prettierrc`

No further conventions are specified.

## Markdown

- Always indicate code type in code blocks as in Python in

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

  - TypeScript
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

### Integration tests

See `tests/integration`.

### Python versions

The project requires Python `^3.10`. Pull request tests run on `3.10`, `3.11`,
`3.12` and `3.13`, with `3.12` as the default job.

Each version runs the `basic`, `optional` and `snowflake` marker suites, with one
exception: the `snowflake` suite is skipped on `3.12`.

### Running tests locally

```shell
> make test-unit
> make test-unit-basic
> make test-unit-optional
```

The `test-<suite>-<marker>` targets are pattern rules, so any suite and marker
combine. Golden files are regenerated with `make write-golden-<name>`.

### Test pipelines

Defined in `.azure_pipelines/ci-eval-pr.yaml`, which calls the shared steps in
`.azure_pipelines/templates/run-tests.yaml`. See
[`.azure_pipelines/README.md`](https://github.com/truera/trulens/blob/main/.azure_pipelines/README.md)
for how the pipelines fit together.
