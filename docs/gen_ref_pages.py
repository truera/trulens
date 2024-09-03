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

pack_symbol = (
    "📦"  # <code class="doc-symbol doc-symbol-nav doc-symbol-package"></code>'
)


_SPECIAL_FORMATTING = {
    "litellm": f"{pack_symbol} LiteLLM",
    "openai": f"{pack_symbol} OpenAI",
    "huggingface": f"{pack_symbol} HuggingFace",
    "langchain": f"{pack_symbol} LangChain",
    "llamaindex": f"{pack_symbol} LlamaIndex",
    "nemo": f"{pack_symbol} Nemo Guardrails",
    "cortex": f"{pack_symbol} Snowflake Cortex",
    "bedrock": f"{pack_symbol} Amazon Bedrock",
    "snowflake": f"{pack_symbol} Snowflake",
    "basic": f"{mod_symbol} basic",
    "custom": f"{mod_symbol} custom",
    "virtual": f"{mod_symbol} virtual",
}


def format_parts(parts: Tuple[str, ...]) -> Tuple[str, ...]:
    if parts[0] == "trulens":
        parts = tuple(parts[1:])

    external_package = len(parts) and parts[0] in (
        "connectors",
        "providers",
        "apps",
    )
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
            # return _SPECIAL_FORMATTING[part]
            return _SPECIAL_FORMATTING[part]

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

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        return

    full_doc_path = docs_reference_path / doc_path
    nav_parts = format_parts(parts)
    nav[nav_parts] = doc_path.as_posix()

    # if (
    #    parts[0] == "trulens_eval"
    # ):  # legacy module is in the trulens-legacy package
    #    nav_parts = format_parts(("legacy", *parts))
    # else:
    # nav_parts = format_parts(parts)
    #    print(nav_parts)

    if not content:
        ident = ".".join(parts)
        content = f"# {ident}\n::: {ident}"

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
app_packages = [f"apps/{pkg_dir}" for pkg_dir in next(os.walk("src/apps"))[1]]
connector_packages = [
    f"connectors/{pkg_dir}" for pkg_dir in next(os.walk("src/connectors"))[1]
]
legacy_packages = ["trulens_eval"]
packages = (
    core_packages + provider_packages + app_packages + connector_packages
    #     + legacy_packages # don't generate these as they seem to be blank anyway
)
print("Collecting from packages:", packages)

nav["API Reference"] = "index.md"
nav["providers"] = "providers/index.md"
nav["apps"] = "apps/index.md"
nav["connectors"] = "connectors/index.md"
nav["❌ trulens_eval"] = "trulens_eval/index.md"

for package in packages:
    # create a nav entry for package/index.md
    package_path = src / package
    for path in sorted(package_path.rglob("*.py")):
        # ignore .venv under package_path
        if package_path / ".venv" in path.parents:
            continue

        module_path = path.relative_to(package_path).with_suffix("")
        doc_path = path.relative_to(package_path).with_suffix(".md")
        parts = tuple(module_path.parts)

        if any(
            (part.startswith("_") and not part.endswith("_")) for part in parts
        ):
            # skip private modules
            continue

        if not len(parts):
            continue

        elif package == "trulens_eval":
            pass
            # if parts[0] == "trulens_eval":
            #    parts = ("legacy", *parts[1:])
        elif package == "dashboard":
            if parts[0] != "trulens":
                continue
            if len(parts) > 2 and parts[2] == "react_components":
                continue
        elif path.parent.name == "apps" and path.name in [
            "basic.py",
            "custom.py",
            "virtual.py",
        ]:
            # Write core apps. Pass to skip the next condition.
            pass
        elif not os.path.exists(path.parent / "__init__.py"):
            print(
                "Skipping due to missing __init__.py: ",
                path.parent / "__init__.py",
            )
            continue

        write_to_gen_files(parts, doc_path=doc_path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
