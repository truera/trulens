#!/usr/bin/env python3
"""Migrate Feedback( → Metric( in all example files."""

import json
import os
from pathlib import Path
import re

BASE = Path(
    "/Users/jreini/.snowflake/cortex/worktree/trulens/phase5-examples/examples"
)

CHANGED_FILES = []
ISSUES = []


def migrate_source_code(code: str, filepath: str) -> str:
    original = code

    code = migrate_imports(code, filepath)
    code = migrate_type_annotations(code, filepath)
    code = migrate_feedback_calls(code, filepath)

    if code != original:
        CHANGED_FILES.append(filepath)

    return code


def migrate_imports(code: str, filepath: str) -> str:
    # Pattern: from trulens.core import Feedback
    # Also handle: from trulens.core import Feedback, TruSession etc.
    # Also handle: from trulens.core.feedback import feedback as core_feedback

    # Case 1: standalone `from trulens.core import Feedback`
    code = re.sub(
        r"from trulens\.core import Feedback\b",
        "from trulens.core import Metric, Selector",
        code,
    )

    # Case 2: `from trulens.core import Feedback, X, Y` or `from trulens.core import X, Feedback, Y`
    # Replace Feedback with Metric within multi-import lines from trulens.core
    def fix_multi_import(m):
        line = m.group(0)
        # Already has Metric? Just remove Feedback
        if "Metric" in line:
            line = re.sub(r",\s*Feedback\b", "", line)
            line = re.sub(r"Feedback\s*,\s*", "", line)
            # Ensure Selector is there
            if "Selector" not in line:
                line = line.rstrip()
                # Add Selector before the closing
                line = re.sub(
                    r"(import\s+.*)",
                    lambda mm: mm.group(1).rstrip() + ", Selector"
                    if "Selector" not in mm.group(1)
                    else mm.group(1),
                    line,
                )
            return line
        # Replace Feedback with Metric, Selector
        if "Selector" in line:
            line = re.sub(r"\bFeedback\b", "Metric", line)
        else:
            line = re.sub(r"\bFeedback\b", "Metric, Selector", line)
        return line

    code = re.sub(
        r"from trulens\.core import [^\n]*\bFeedback\b[^\n]*",
        fix_multi_import,
        code,
    )

    # Case 3: Remove standalone `from trulens.core.feedback.selector import Selector`
    # since Selector is now imported from trulens.core
    code = re.sub(
        r"from trulens\.core\.feedback\.selector import Selector\n?",
        "",
        code,
    )

    # Case 4: `from trulens.core.feedback import feedback as core_feedback`
    # The actual Feedback class usage like `core_feedback.Feedback(...)` will be handled in migrate_feedback_calls

    return code


def migrate_type_annotations(code: str, filepath: str) -> str:
    code = re.sub(r"(\)\s*->\s*)Feedback\b", r"\1Metric", code)
    code = re.sub(r": Feedback\b", ": Metric", code)
    return code


def migrate_feedback_calls(code: str, filepath: str) -> str:
    # Strategy: We need to find complete Feedback(...).on_*().on_*().aggregate() chains
    # and convert them to Metric(implementation=..., selectors={...}, agg=...)

    # Handle core_feedback.Feedback( pattern (App_TruBot.py style)
    code = re.sub(r"\bcore_feedback\.Feedback\b", "core_feedback.Metric", code)

    # --- PATTERN A: Feedback(...).on({...selectors...}) already using new Selector style ---
    # These just need Feedback → Metric and .on({...}) → selectors={...}
    # We'll handle these together with the general rewrite

    # --- General approach: regex-based line-by-line rewrite ---
    code = rewrite_feedback_chains(code, filepath)

    return code


def rewrite_feedback_chains(code: str, filepath: str) -> str:
    """Rewrite all Feedback(...) chains to Metric(...) form."""

    # We process the code as a single string, finding Feedback( occurrences
    # and rewriting them with their chained .on_*() and .aggregate() calls.

    result = []
    i = 0
    while i < len(code):
        # Find next Feedback( occurrence
        match = re.search(r"\bFeedback\s*\(", code[i:])
        if not match:
            result.append(code[i:])
            break

        # Append everything before this match
        result.append(code[i : i + match.start()])
        pos = i + match.start()

        # Try to parse and rewrite this Feedback chain
        try:
            new_text, end_pos = parse_and_rewrite_feedback(code, pos, filepath)
            result.append(new_text)
            i = end_pos
        except Exception as e:
            ISSUES.append(
                f"{filepath}: Failed to parse Feedback at pos {pos}: {e}"
            )
            result.append("Feedback(")
            i = pos + match.end() - match.start()

    return "".join(result)


def find_matching_paren(code: str, start: int) -> int:
    """Find matching closing paren for opening paren at start."""
    depth = 0
    in_string = None
    escape = False
    i = start
    while i < len(code):
        c = code[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == "\\":
            escape = True
            i += 1
            continue
        if in_string:
            if c == in_string:
                # Check for triple quotes
                if code[i : i + 3] == in_string * 3:
                    # End triple quote
                    i += 3
                    in_string = None
                    continue
                elif i >= 2 and code[i - 2 : i + 1] == in_string * 3:
                    i += 1
                    continue
                else:
                    in_string = None
            i += 1
            continue
        if c in ('"', "'"):
            if code[i : i + 3] == c * 3:
                in_string = c
                i += 3
                continue
            else:
                in_string = c
                i += 1
                continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return -1


def parse_and_rewrite_feedback(code: str, pos: int, filepath: str):
    """Parse a Feedback(...) chain and rewrite to Metric(...)."""

    # pos points to 'F' of 'Feedback('
    fb_match = re.match(r"Feedback\s*\(", code[pos:])
    if not fb_match:
        raise ValueError("Not at Feedback(")

    paren_start = pos + fb_match.end() - 1  # position of '('
    paren_end = find_matching_paren(code, paren_start)
    if paren_end == -1:
        raise ValueError("Unmatched paren")

    # Content inside Feedback(...)
    inner = code[paren_start + 1 : paren_end].strip()

    # Parse inner: first arg is the implementation, rest are kwargs
    impl_str, kwargs_str = split_first_arg(inner)

    # Now look for chained calls after the closing paren
    chain_pos = paren_end + 1
    selectors = {}
    agg = None
    extra_kwargs = {}

    # Parse chained .on_*(), .on({...}), .aggregate() calls
    while chain_pos < len(code):
        # Skip whitespace and newlines
        ws_match = re.match(r"\s*", code[chain_pos:])
        if ws_match:
            chain_pos += ws_match.end()

        # Check for .on_input_output()
        m = re.match(r"\.on_input_output\s*\(\s*\)", code[chain_pos:])
        if m:
            selectors["prompt"] = "Selector.select_record_input()"
            selectors["response"] = "Selector.select_record_output()"
            chain_pos += m.end()
            continue

        # Check for .on_input()
        m = re.match(r"\.on_input\s*\(\s*\)", code[chain_pos:])
        if m:
            selectors["prompt"] = "Selector.select_record_input()"
            chain_pos += m.end()
            continue

        # Check for .on_output()
        m = re.match(r"\.on_output\s*\(\s*\)", code[chain_pos:])
        if m:
            selectors["response"] = "Selector.select_record_output()"
            chain_pos += m.end()
            continue

        # Check for .on_context(collect_list=True/False)
        m = re.match(
            r"\.on_context\s*\(\s*collect_list\s*=\s*(True|False)\s*\)",
            code[chain_pos:],
        )
        if m:
            collect_val = m.group(1)
            if collect_val == "True":
                selectors["source"] = (
                    f"Selector.select_context(collect_list={collect_val})"
                )
            else:
                selectors["context"] = (
                    f"Selector.select_context(collect_list={collect_val})"
                )
            chain_pos += m.end()
            continue

        # Check for .on_context() without args
        m = re.match(r"\.on_context\s*\(\s*\)", code[chain_pos:])
        if m:
            selectors["context"] = "Selector.select_context()"
            chain_pos += m.end()
            continue

        # Check for .on({...}) with Selector-based dict
        m = re.match(r"\.on\s*\(\s*\{", code[chain_pos:])
        if m:
            # Find the matching } then )
            brace_start = chain_pos + m.end() - 1
            brace_end = find_matching_paren(code, brace_start)
            if brace_end == -1:
                break
            # Find the closing )
            close_paren = code.index(")", brace_end + 1)
            dict_content = code[brace_start : brace_end + 1].strip()
            # Parse the dict entries and add to selectors
            # We'll keep them as-is
            entries = parse_dict_entries(dict_content)
            selectors.update(entries)
            chain_pos = close_paren + 1
            continue

        # Check for .on(trace_selector) or .on(some_var)
        m = re.match(r"\.on\s*\(", code[chain_pos:])
        if m:
            on_paren_start = chain_pos + m.end() - 1
            on_paren_end = find_matching_paren(code, on_paren_start)
            if on_paren_end == -1:
                break
            on_content = code[on_paren_start + 1 : on_paren_end].strip()

            # Check if it's a dict literal
            if on_content.startswith("{"):
                entries = parse_dict_entries(on_content)
                selectors.update(entries)
            elif "=" in on_content and not on_content.startswith("text="):
                # keyword args like: prompt=X, response=Y
                # This is an old .on(prompt=Select.RecordInput, ...) pattern
                for kv in split_kwargs(on_content):
                    k, v = kv.split("=", 1)
                    selectors[k.strip()] = migrate_old_select(v.strip())
            elif on_content.strip() in ("trace_selector",):
                # Variable reference that itself is a dict
                selectors["__var__"] = on_content.strip()
            else:
                # Some other .on() pattern - old-style selectors
                # e.g., .on(context.collect()), .on(context), .on(text=Select.Record.main_output[::20])
                # Handle .on(\ntext=...\n) multiline
                kw_match = re.match(r"(\w+)\s*=\s*(.*)", on_content, re.DOTALL)
                if kw_match:
                    k = kw_match.group(1).strip()
                    v = kw_match.group(2).strip()
                    selectors[k] = migrate_old_select(v)
                else:
                    # Positional .on(context) etc - keep as-is but flag
                    selectors["__positional__"] = on_content.strip()

            chain_pos = on_paren_end + 1
            continue

        # Check for .aggregate(...)
        m = re.match(r"\.aggregate\s*\(", code[chain_pos:])
        if m:
            agg_paren_start = chain_pos + m.end() - 1
            agg_paren_end = find_matching_paren(code, agg_paren_start)
            if agg_paren_end == -1:
                break
            agg = code[agg_paren_start + 1 : agg_paren_end].strip()
            chain_pos = agg_paren_end + 1
            continue

        # No more chained calls
        break

    # Now build the Metric(...) call
    new_code = build_metric_call(impl_str, kwargs_str, selectors, agg, filepath)

    return new_code, chain_pos


def split_first_arg(inner: str):
    """Split 'fn, name=\"X\", criteria=\"Y\"' into ('fn', 'name=\"X\", criteria=\"Y\"')."""
    depth = 0
    in_string = None
    escape = False
    for i, c in enumerate(inner):
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if in_string:
            if c == in_string:
                in_string = None
            continue
        if c in ('"', "'"):
            in_string = c
            continue
        if c in ("(", "[", "{"):
            depth += 1
        elif c in (")", "]", "}"):
            depth -= 1
        elif c == "," and depth == 0:
            first = inner[:i].strip()
            rest = inner[i + 1 :].strip()
            return first, rest
    return inner.strip(), ""


def split_kwargs(s: str):
    """Split kwargs string respecting nesting."""
    result = []
    depth = 0
    current = []
    in_string = None
    for c in s:
        if in_string:
            current.append(c)
            if c == in_string:
                in_string = None
            continue
        if c in ('"', "'"):
            in_string = c
            current.append(c)
            continue
        if c in ("(", "[", "{"):
            depth += 1
            current.append(c)
        elif c in (")", "]", "}"):
            depth -= 1
            current.append(c)
        elif c == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(c)
    if current:
        result.append("".join(current).strip())
    return result


def parse_dict_entries(dict_str: str):
    """Parse a dict literal like '{"key": value, ...}' into dict."""
    # Strip outer braces
    inner = dict_str.strip()
    if inner.startswith("{"):
        inner = inner[1:]
    if inner.endswith("}"):
        inner = inner[:-1]
    inner = inner.strip()

    entries = {}
    # Split by commas at depth 0
    parts = split_kwargs(inner)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Split on first :
        colon_pos = find_colon(part)
        if colon_pos == -1:
            continue
        key = part[:colon_pos].strip().strip('"').strip("'")
        val = part[colon_pos + 1 :].strip()
        entries[key] = val
    return entries


def find_colon(s: str) -> int:
    """Find first colon not inside string or nested structure."""
    depth = 0
    in_string = None
    for i, c in enumerate(s):
        if in_string:
            if c == in_string:
                in_string = None
            continue
        if c in ('"', "'"):
            in_string = c
            continue
        if c in ("(", "[", "{"):
            depth += 1
        elif c in (")", "]", "}"):
            depth -= 1
        elif c == ":" and depth == 0:
            return i
    return -1


def migrate_old_select(val: str) -> str:
    """Migrate old Select.RecordInput etc. to new Selector.select_*."""
    val = val.strip()
    if val in ("select_schema.Select.RecordInput", "Select.RecordInput"):
        return "Selector.select_record_input()"
    if val in ("select_schema.Select.RecordOutput", "Select.RecordOutput"):
        return "Selector.select_record_output()"
    # Keep complex selectors as-is (they may be old-style paths)
    return val


def build_metric_call(impl_str, kwargs_str, selectors, agg, filepath):
    """Build a Metric(...) call string."""
    parts = [f"implementation={impl_str}"]

    # Add kwargs (name=, criteria=, etc.)
    if kwargs_str:
        parts.append(kwargs_str)

    # Add selectors
    if selectors:
        # Check for special cases
        if "__var__" in selectors:
            var_name = selectors.pop("__var__")
            if selectors:
                sel_str = format_selectors(selectors)
                parts.append(f"selectors={{**{var_name}, {sel_str[1:-1]}}}")
            else:
                parts.append(f"selectors={var_name}")
        elif "__positional__" in selectors:
            # Old-style positional .on() - keep as comment/flag
            pos_val = selectors.pop("__positional__")
            if selectors:
                sel_str = format_selectors(selectors)
                ISSUES.append(
                    f"{filepath}: Positional .on({pos_val}) needs manual review"
                )
                parts.append(f"selectors={sel_str}")
            else:
                ISSUES.append(
                    f"{filepath}: Positional .on({pos_val}) needs manual review"
                )
        else:
            sel_str = format_selectors(selectors)
            parts.append(f"selectors={sel_str}")

    # Add aggregation
    if agg:
        parts.append(f"agg={agg}")

    # Format
    inner = ", ".join(parts)

    # If it's short enough, single line
    if len(inner) < 80:
        return f"Metric({inner})"

    # Multi-line
    indent = "    "
    lines = ["Metric(\n"]
    for p in parts:
        lines.append(f"{indent}{p},\n")
    lines.append(")")
    return "".join(lines)


def format_selectors(selectors: dict) -> str:
    """Format selectors dict."""
    items = []
    for k, v in selectors.items():
        items.append(f'"{k}": {v}')
    if len(items) == 1:
        return "{" + items[0] + "}"
    inner = ", ".join(items)
    if len(inner) < 60:
        return "{" + inner + "}"
    # Multi-line
    lines = ["{\n"]
    for item in items:
        lines.append(f"        {item},\n")
    lines.append("    }")
    return "".join(lines)


def process_notebook(filepath: str):
    """Process a .ipynb notebook file."""
    if not os.path.isfile(filepath):
        return
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            nb = json.load(f)
        except json.JSONDecodeError:
            ISSUES.append(f"{filepath}: Invalid JSON")
            return

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source_lines = cell.get("source", [])
        if not source_lines:
            continue

        source = "".join(source_lines)
        if "Feedback(" not in source and "Feedback" not in source:
            continue

        new_source = migrate_source_code(source, filepath)
        if new_source != source:
            changed = True
            # Split back into lines preserving notebook format
            new_lines = split_to_notebook_lines(new_source)
            cell["source"] = new_lines

    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")


def split_to_notebook_lines(code: str) -> list:
    """Split code string back into notebook source line format."""
    lines = code.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            if line:  # Don't add empty trailing line
                result.append(line)
    return result


def process_py_file(filepath: str):
    """Process a .py file."""
    if not os.path.isfile(filepath):
        return
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    if "Feedback(" not in code and "Feedback" not in code:
        return

    new_code = migrate_source_code(code, filepath)
    if new_code != code:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_code)


def main():
    files = []
    for ext in ("*.ipynb", "*.py"):
        files.extend(BASE.rglob(ext))

    files.sort()
    print(f"Found {len(files)} total files to scan")

    for f in files:
        filepath = str(f)
        if ".ipynb_checkpoints" in filepath:
            continue
        if filepath.endswith(".ipynb"):
            process_notebook(filepath)
        elif filepath.endswith(".py"):
            process_py_file(filepath)

    print(f"\n{'=' * 60}")
    print("MIGRATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Files changed: {len(CHANGED_FILES)}")
    for f in sorted(set(CHANGED_FILES)):
        print(f"  {f}")
    print(f"\nIssues ({len(ISSUES)}):")
    for issue in ISSUES:
        print(f"  {issue}")


if __name__ == "__main__":
    main()
