"""Generate the code reference pages and navigation."""

import os
from pathlib import Path
from typing import Optional, Tuple

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "src"

docs_reference_path = Path("reference")

mod_symbol = '<code class="doc-symbol doc-symbol-nav doc-symbol-module"></code>'

pack_symbol = "(package)"  # <code class="doc-symbol doc-symbol-nav doc-symbol-package"></code>'


_SPECIAL_FORMATTING = {
    "litellm": "LiteLLM",
    "openai": "OpenAI",
    "huggingface": "HuggingFace",
    "langchain": "LangChain",
    "llamaindex": "LlamaIndex",
    "nemo": "Nemo Guardrails",
    "cortex": "Snowflake Cortex",
    "bedrock": "Amazon Bedrock",
}


def format_parts(parts: Tuple[str, ...]) -> Tuple[str, ...]:
    if parts[0] == "trulens":
        parts = tuple(parts[1:])

    external_package = len(parts) and parts[0] in ("providers", "instrument")
    seen_package_level = False

    def _format_part(idx: int, part: str):
        nonlocal seen_package_level
        if not external_package and idx == 1:
            # internal package path starts at 1
            # trulens.core.utils -> core is nonmodule, utils is module
            seen_package_level = True

        if external_package and idx == 1 and part in _SPECIAL_FORMATTING:
            # trulens.providers.langchain.provider -> providers is nonmodule, langchain is package-level, provider is module
            seen_package_level = True
            return pack_symbol + " " + _SPECIAL_FORMATTING[part]

        if not seen_package_level:
            return part

        return f"{mod_symbol} {part}"

    return tuple(_format_part(i, part) for i, part in enumerate(parts))


def write_to_gen_files(
    parts: Tuple[str, ...],
    content: Optional[str] = None,
    doc_path: Optional[Path] = None,
):
    doc_path = doc_path or Path(*parts)

    print(parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")

    elif parts[-1] == "__main__":
        return

    full_doc_path = docs_reference_path / doc_path

    if (
        parts[0] == "trulens_eval"
    ):  # legacy module is in the trulens-legacy package
        nav_parts = format_parts(("legacy", *parts))
    else:
        nav_parts = format_parts(parts)

    #    print(nav_parts)

    nav[nav_parts] = doc_path.as_posix()

    if not content:
        ident = ".".join(parts)
        content = f"# {ident}\n::: {ident}"

        if "legacy" in parts:
            # Show fewer details in the legacy sections.
            content += """
    options:
        heading_level: 2
        show_bases: false
        show_root_heading: false
        show_root_toc_entry: false
        show_source: false
        show_docstring_classes: false
        show_docstring_modules: false
        show_docstring_parameters: false
        show_docstring_returns: false
        show_docstring_description: true
        show_docstring_examples: false
        show_docstring_other_parameters: false
        show_docstring_attributes: false
        show_signature: false
        separate_signature: false
        summary: false
        group_by_category: true
        members_order: alphabetical
"""
        if "api" in parts:
            # Show fewer details in the API sections.
            content += """
    options:
        heading_level: 2
        show_bases: false
        show_root_heading: false
        show_root_toc_entry: false
        show_source: false
        show_docstring_classes: false
        show_docstring_modules: false
        show_docstring_parameters: false
        show_docstring_returns: false
        show_docstring_description: true
        show_docstring_examples: false
        show_docstring_other_parameters: false
        show_docstring_attributes: false
        show_signature: false
        separate_signature: false
        summary: false
        group_by_category: true
        members_order: alphabetical
"""
    #            content = f"API {ident} [{ident}][{ident}]"

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        print("writing to", full_doc_path)
        fd.write(content)


core_packages = [
    "api",
    "legacy",
    "core",
    "feedback",
    "dashboard",
    "benchmark",
]
provider_packages = [
    f"providers/{pkg_dir}" for pkg_dir in next(os.walk("src/providers"))[1]
]
instrument_packages = [
    f"instrument/{pkg_dir}" for pkg_dir in next(os.walk("src/instrument"))[1]
]
packages = core_packages + provider_packages + instrument_packages
print("Collecting from packages:", packages)

nav["API Reference"] = "index.md"
nav["providers"] = "providers/index.md"
nav["instrument"] = "instrument/index.md"
nav["legacy"] = "legacy/index.md"

for package in packages:
    # create a nav entry for package/index.md
    package_path = src / package
    for path in sorted(package_path.rglob("*.py")):
        module_path = path.relative_to(package_path).with_suffix("")
        doc_path = path.relative_to(package_path).with_suffix(".md")
        parts = tuple(module_path.parts)

        if not len(parts):
            continue
        if package == "dashboard":
            if parts[0] != "trulens":
                continue
            if len(parts) > 2 and parts[2] == "react_components":
                continue
        if not os.path.exists(path.parent / "__init__.py"):
            print(
                "Skipping due to missing python module: ",
                path.parent / "__init__.py",
            )
            continue

        write_to_gen_files(parts, doc_path=doc_path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

# with mkdocs_gen_files.open("reference/index.md", "w") as nav_file:
#    nav_file.writelines(nav.build_literate_nav())
