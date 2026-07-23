"""Integration test for prompt-template rendering across all feedback templates.

Addresses truera/trulens#2493.

TruLens ships 20+ feedback template classes (in ``rag.py``, ``safety.py``,
``quality.py``, ``agent.py``) that expose ``system_prompt`` /
``system_prompt_template`` / ``user_prompt`` / ``criteria`` / ... strings with
``{placeholder}`` fields. Providers build the LLM evaluation prompts by
``str.format``-ing these. Nothing currently verifies that every exported
template actually renders -- i.e. that its braces are well-formed and that
supplying its declared fields leaves no unresolved ``{placeholder}`` behind.

This test enumerates every prompt-template string reachable from
``trulens.feedback.templates.__all__`` and renders each one, which surfaces:

* malformed / unescaped braces (``str.format`` raises), and
* fields that cannot be satisfied / stray placeholders left in the output.
"""

import inspect
import re
from string import Formatter

import pytest
from trulens.feedback import templates as fb_templates

# ClassVar string attributes that hold renderable prompt templates.
_TEMPLATE_STR_ATTRS = (
    "system_prompt",
    "system_prompt_template",
    "user_prompt",
    "criteria",
    "criteria_template",
    "output_space_prompt",
    "default_cot_prompt",
    "sentences_splitter_prompt",
    "prompt",
)

# Matches a leftover ``{field_name}`` placeholder (not rendered JSON like
# ``{"score": 2}``, whose first char after ``{`` is not an identifier char).
_UNRESOLVED_PLACEHOLDER = re.compile(r"\{[A-Za-z_]\w*\}")


def _collect_template_strings():
    """Return ``(id, template_str)`` for every prompt-template ClassVar on
    every class exported from ``trulens.feedback.templates.__all__``."""
    cases = []
    for name in fb_templates.__all__:
        obj = getattr(fb_templates, name, None)
        if not inspect.isclass(obj):
            continue
        for attr in _TEMPLATE_STR_ATTRS:
            value = getattr(obj, attr, None)
            if isinstance(value, str) and value.strip():
                cases.append((f"{name}.{attr}", value))
    return cases


_TEMPLATE_CASES = _collect_template_strings()


def test_templates_were_discovered():
    """Guard so the parametrization below can't pass vacuously."""
    assert (
        len(_TEMPLATE_CASES) > 10
    ), f"expected to discover many templates, found {len(_TEMPLATE_CASES)}"


@pytest.mark.parametrize(
    "template",
    [case[1] for case in _TEMPLATE_CASES],
    ids=[case[0] for case in _TEMPLATE_CASES],
)
def test_template_renders_without_unresolved_placeholders(template):
    """Every exported template renders cleanly once its fields are supplied."""
    declared_fields = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name
    }

    # Supplying every declared field must fully render the template. This also
    # surfaces malformed/unescaped braces, which raise inside ``str.format``.
    rendered = template.format(**{f: f"<{f}>" for f in declared_fields})

    assert rendered.strip(), "template rendered to an empty string"
    leftover = _UNRESOLVED_PLACEHOLDER.findall(rendered)
    assert (
        not leftover
    ), f"unresolved placeholders remain after render: {leftover}"
