import pandas as pd
from trulens.core.schema import prompt as prompt_schema
from trulens.dashboard.tabs.Prompts import _preview
from trulens.dashboard.tabs.Prompts import _structured_diff
from trulens.dashboard.tabs.Prompts import _version_label_map
from trulens.dashboard.tabs.Prompts import _version_lines
from trulens.dashboard.tabs.Prompts import _version_options


def _prompt() -> prompt_schema.Prompt:
    return prompt_schema.Prompt(slug="support-assistant", prompt_type="chat")


def _version(system: str) -> prompt_schema.PromptVersion:
    return prompt_schema.PromptVersion(
        prompt_id=_prompt().prompt_id,
        prompt_type="chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "{{question}}"},
        ],
        variables=["question"],
    )


def _labels() -> pd.DataFrame:
    return pd.DataFrame({
        "prompt_id": ["p", "p"],
        "label": ["production", "latest"],
        "version_id": ["v1", "v2"],
        "updated_at": [None, None],
    })


def _versions() -> pd.DataFrame:
    return pd.DataFrame({
        "version_id": ["v1", "v2"],
        "change_note": ["baseline", None],
    })


def test_version_label_map_groups_and_sorts_labels():
    labels = pd.concat([
        _labels(),
        pd.DataFrame([
            {
                "prompt_id": "p",
                "label": "canary",
                "version_id": "v1",
                "updated_at": None,
            }
        ]),
    ])

    assert _version_label_map(labels) == {
        "v1": ["canary", "production"],
        "v2": ["latest"],
    }


def test_version_label_map_handles_no_labels():
    assert _version_label_map(pd.DataFrame()) == {}


def test_version_options_annotate_labels_and_missing_notes():
    options = _version_options(_versions(), _labels())

    assert options["v1"] == "v1… baseline [production]"
    assert options["v2"] == "v2… no change note [latest]"


def test_version_lines_tag_each_role():
    lines = _version_lines(_version("Be brief."))

    assert lines == ["[system]", "Be brief.", "[user]", "{{question}}"]


def test_structured_diff_marks_exactly_the_changed_message():
    diff = _structured_diff(
        _version("Be brief."), _version("Be thorough.")
    ).splitlines()

    removed = [
        line
        for line in diff
        if line.startswith("-") and not line.startswith("---")
    ]
    added = [
        line
        for line in diff
        if line.startswith("+") and not line.startswith("+++")
    ]

    assert removed == ["-Be brief."]
    assert added == ["+Be thorough."]


def test_structured_diff_is_empty_for_identical_content():
    assert _structured_diff(_version("Be brief."), _version("Be brief.")) == ""


def test_preview_renders_locally():
    preview, error = _preview(
        _version("Be brief."), _prompt(), {"question": "why"}
    )

    assert error is None
    assert "why" in preview


def test_preview_reports_missing_variables_instead_of_raising():
    preview, error = _preview(_version("Be brief."), _prompt(), {})

    assert preview is None
    assert "question" in error
