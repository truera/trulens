"""Generate the code reference pages and navigation."""

from inspect import cleandoc
import os
from pathlib import Path
from typing import Optional

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "src"

docs_reference_path = Path("reference")

mod_symbol = '<code class="doc-symbol doc-symbol-nav doc-symbol-module"></code>'


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


def format_parts(parts: tuple):
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
            return _SPECIAL_FORMATTING[part]
        if not seen_package_level:
            return part
        return f"{mod_symbol} {part}"

    return tuple(_format_part(i, part) for i, part in enumerate(parts))


def write_to_gen_files(
    parts: tuple, content: Optional[str] = None, doc_path: Optional[Path] = None
):
    doc_path = doc_path or Path(*parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        return

    full_doc_path = docs_reference_path / doc_path
    nav_parts = format_parts(parts)
    nav[nav_parts] = doc_path.as_posix()

    if not content:
        ident = ".".join(parts)
        content = f"#{ident}\n::: {ident}"

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(content)


core_packages = [
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

# Write Index Page
with mkdocs_gen_files.open(
    docs_reference_path / "trulens" / "index.md", "w"
) as fd:
    fd.write(
        cleandoc(
            """
            # TruLens API Reference

            Welcome to the TruLens API Reference!
            Use the search and navigation to explore the various modules and classes available in the TruLens library.
            """
        )
    )

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
                "Skipping due to missing python package: ",
                path.parent / "__init__.py",
            )
            continue

        write_to_gen_files(parts, doc_path=doc_path)

    with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())
