#!/usr/bin/env python3
"""Migrate Feedback( API to Metric( API in cookbook files."""

import json
import os
import re
import sys

COOKBOOK_DIR = os.path.join(os.path.dirname(__file__), "docs", "cookbook")


def find_files_with_feedback(root):
    """Find all .ipynb and .py files containing Feedback("""
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith((".ipynb", ".py")):
                fpath = os.path.join(dirpath, fn)
                with open(fpath, "r") as f:
                    content = f.read()
                if "Feedback(" in content:
                    matches.append(fpath)
    return sorted(matches)


def migrate_source(source: str) -> str:
    """Migrate a single source string (cell source or .py file content)."""

    # 1. Fix imports: from trulens.core import Feedback -> from trulens.core import Metric, Selector
    # Handle various import patterns

    # Pattern: from trulens.core import Feedback, TruSession (and other combos)
    def fix_import_line(line):
        if "from trulens.core import" in line and "Feedback" in line:
            # Don't touch lines that already have Metric
            if "Metric" in line:
                return line
            # Replace Feedback with Metric
            line = line.replace("Feedback", "Metric")
            # Add Selector if not already present
            if "Selector" not in line:
                # Find the import list and add Selector
                # Handle multi-item imports
                line = line.rstrip()
                if line.endswith(")"):
                    line = line[:-1] + ", Selector)"
                else:
                    line = line + ", Selector"
            return line
        return line

    lines = source.split("\n")
    new_lines = []
    for line in lines:
        new_lines.append(fix_import_line(line))
    source = "\n".join(new_lines)

    # Also handle: from trulens.core import Feedback\n (in notebook JSON with \n)
    # This is handled above since we split on \n

    # Remove standalone "from trulens.core.feedback.selector import Selector" if we already added it to the core import
    # Only remove if we have "from trulens.core import Metric, Selector" or similar
    if (
        "from trulens.core import" in source
        and "Selector"
        in source.split("from trulens.core import")[1].split("\n")[0]
    ):
        # Check if there's a separate Selector import that's now redundant
        source = re.sub(
            r"\nfrom trulens\.core\.feedback\.selector import Selector\n",
            "\n",
            source,
        )
        source = re.sub(
            r"^from trulens\.core\.feedback\.selector import Selector\n",
            "",
            source,
        )

    # 2. Migrate Feedback( patterns to Metric(

    # We need to handle complex multi-line patterns. Let's use a state machine approach.
    source = migrate_feedback_calls(source)

    return source


def migrate_feedback_calls(source: str) -> str:
    """Migrate all Feedback( calls to Metric( calls in source."""

    # Strategy: find each Feedback( occurrence and parse the full expression
    # including chained .on_input(), .on_output(), .on_context(), .on(), .aggregate()

    result = []
    i = 0
    while i < len(source):
        # Look for Feedback( but not in comments or strings that are about the class name
        # Match: Feedback( at word boundary but not "class Feedback" or "def Feedback"
        # Also skip "-> Feedback:" type annotations
        match = re.search(r"(?<!\w)Feedback\(", source[i:])
        if not match:
            result.append(source[i:])
            break

        start = i + match.start()
        result.append(source[i:start])

        # Check if this is a type annotation like "-> Feedback:" or "Feedback[" or "class Feedback"
        # Look at what comes before
        before = source[max(0, start - 20) : start].strip()
        if (
            before.endswith("->")
            or before.endswith("class")
            or before.endswith("def")
            or before.endswith(":")
        ):
            result.append("Feedback(")
            i = start + len("Feedback(")
            continue

        # Parse the full Feedback(...) expression with potential chained calls
        expr_start = start
        expr, expr_end = parse_feedback_expression(source, start)

        if expr is None:
            # Couldn't parse, skip
            result.append("Feedback(")
            i = start + len("Feedback(")
            continue

        # Convert to Metric
        metric_expr = convert_to_metric(expr)
        if metric_expr is not None:
            result.append(metric_expr)
        else:
            # Fallback: just replace Feedback( with Metric(implementation=
            result.append(source[start:expr_end])
        i = expr_end

    return "".join(result)


def find_matching_paren(source, start):
    """Find the matching closing paren for the opening paren at start."""
    depth = 0
    i = start
    in_string = None
    escape_next = False
    while i < len(source):
        c = source[i]
        if escape_next:
            escape_next = False
            i += 1
            continue
        if c == "\\":
            escape_next = True
            i += 1
            continue
        if in_string:
            if c == in_string:
                # Check for triple quotes
                if source[i : i + 3] == in_string * 3:
                    in_string = None
                    i += 3
                    continue
                in_string = None
            i += 1
            continue
        if c in ('"', "'"):
            if source[i : i + 3] in ('"""', "'''"):
                in_string = c
                i += 3
                continue
            in_string = c
            i += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def parse_feedback_expression(source, start):
    """Parse a Feedback(...).on_xxx().on_yyy().aggregate() expression.

    Returns (parsed_dict, end_index) or (None, None)
    """
    # start points to 'F' in 'Feedback('
    paren_start = start + len("Feedback")
    paren_end = find_matching_paren(source, paren_start)
    if paren_end is None:
        return None, None

    inner = source[paren_start + 1 : paren_end].strip()

    # Parse the inner arguments of Feedback(...)
    expr = {
        "implementation": None,
        "name": None,
        "higher_is_better": None,
        "criteria": None,
        "extra_kwargs": {},
        "chains": [],
        "aggregate": None,
        "raw_inner": inner,
    }

    # Extract the first positional arg (implementation)
    # and named args (name=, higher_is_better=, criteria=)
    impl, named = parse_feedback_args(inner)
    expr["implementation"] = impl
    expr["name"] = named.get("name")
    expr["higher_is_better"] = named.get("higher_is_better")
    expr["criteria"] = named.get("criteria")
    for k, v in named.items():
        if k not in ("name", "higher_is_better", "criteria"):
            expr["extra_kwargs"][k] = v

    # Now parse chained calls after the closing paren
    i = paren_end + 1
    while i < len(source):
        # Skip whitespace and newlines
        j = i
        while j < len(source) and source[j] in (" ", "\t", "\n", "\r"):
            j += 1

        if j < len(source) and source[j] == ".":
            # Parse method call
            method_match = re.match(r"\.(\w+)\s*\(", source[j:])
            if method_match:
                method_name = method_match.group(1)
                method_paren_start = j + method_match.end() - 1
                method_paren_end = find_matching_paren(
                    source, method_paren_start
                )
                if method_paren_end is None:
                    break
                method_args = source[
                    method_paren_start + 1 : method_paren_end
                ].strip()

                if method_name == "aggregate":
                    expr["aggregate"] = method_args
                else:
                    expr["chains"].append((method_name, method_args))

                i = method_paren_end + 1
                continue
        break

    return expr, i


def parse_feedback_args(inner):
    """Parse Feedback() inner arguments.
    Returns (implementation_str, {named_args})
    """
    # Simple approach: find the first arg (up to first comma not inside parens/brackets/strings)
    # then parse remaining as keyword args
    parts = split_args(inner)
    if not parts:
        return "", {}

    impl = parts[0].strip()
    named = {}

    for part in parts[1:]:
        part = part.strip()
        eq_match = re.match(r"(\w+)\s*=\s*", part)
        if eq_match:
            key = eq_match.group(1)
            val = part[eq_match.end() :].strip()
            named[key] = val
        else:
            # Positional arg after first - unusual but keep it
            pass

    return impl, named


def split_args(s):
    """Split a string by commas, respecting parens, brackets, strings."""
    parts = []
    depth = 0
    current = []
    in_string = None
    escape_next = False
    i = 0
    while i < len(s):
        c = s[i]
        if escape_next:
            escape_next = False
            current.append(c)
            i += 1
            continue
        if c == "\\":
            escape_next = True
            current.append(c)
            i += 1
            continue
        if in_string:
            current.append(c)
            if c == in_string:
                in_string = None
            i += 1
            continue
        if c in ('"', "'"):
            in_string = c
            current.append(c)
            i += 1
            continue
        if c in ("(", "[", "{"):
            depth += 1
            current.append(c)
        elif c in (")", "]", "}"):
            depth -= 1
            current.append(c)
        elif c == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(c)
        i += 1
    if current:
        parts.append("".join(current))
    return parts


def convert_to_metric(expr):
    """Convert a parsed Feedback expression to Metric expression string."""
    impl = expr["implementation"]
    if not impl:
        return None

    selectors = {}
    agg = expr.get("aggregate")

    # Process chains
    for method_name, method_args in expr["chains"]:
        if method_name == "on_input":
            selectors["prompt"] = "Selector.select_record_input()"
        elif method_name == "on_output":
            selectors["response"] = "Selector.select_record_output()"
        elif method_name == "on_input_output":
            selectors["prompt"] = "Selector.select_record_input()"
            selectors["response"] = "Selector.select_record_output()"
        elif method_name == "on_context":
            # Parse collect_list arg
            collect_list = "True"  # default
            if method_args:
                cl_match = re.search(
                    r"collect_list\s*=\s*(True|False)", method_args
                )
                if cl_match:
                    collect_list = cl_match.group(1)
            if collect_list == "True":
                selectors["source"] = (
                    "Selector.select_context(collect_list=True)"
                )
            else:
                selectors["context"] = (
                    "Selector.select_context(collect_list=False)"
                )
        elif method_name == "on":
            # .on({...}) - explicit selector dict
            # Parse the dict argument
            args_stripped = method_args.strip()
            if (
                args_stripped.startswith("{")
                or args_stripped.startswith("trace_selector")
                or args_stripped.startswith("selector")
            ):
                # It's a dict or variable - extract key-value pairs
                # For variable references like trace_selector, keep as-is in selectors
                if args_stripped.startswith("{"):
                    # Extract dict contents: {"key": Selector(...)}
                    dict_inner = extract_dict_pairs(args_stripped)
                    for k, v in dict_inner.items():
                        selectors[k] = v
                else:
                    # Variable reference - we'll merge it
                    selectors["__var__"] = args_stripped

    # If only .on_output() and nothing else, use "response" key
    # If only .on_input() and nothing else, use "prompt" key
    # These are already handled above

    # Build the Metric expression
    parts = []
    parts.append(f"implementation={impl}")

    if expr.get("name"):
        parts.append(f"name={expr['name']}")

    if expr.get("higher_is_better") is not None:
        parts.append(f"higher_is_better={expr['higher_is_better']}")

    if expr.get("criteria") is not None:
        parts.append(f"criteria={expr['criteria']}")

    for k, v in expr.get("extra_kwargs", {}).items():
        parts.append(f"{k}={v}")

    if selectors:
        if "__var__" in selectors:
            # Variable reference - use it directly
            var_name = selectors.pop("__var__")
            if selectors:
                # Merge: {**var, "key": Selector...}
                sel_parts = [f"**{var_name}"]
                for k, v in selectors.items():
                    sel_parts.append(f'"{k}": {v}')
                parts.append("selectors={" + ", ".join(sel_parts) + "}")
            else:
                parts.append(f"selectors={var_name}")
        else:
            sel_items = []
            for k, v in selectors.items():
                sel_items.append(f'"{k}": {v}')
            sel_str = "{\n        " + ",\n        ".join(sel_items) + ",\n    }"
            if len(sel_items) == 1:
                sel_str = "{" + sel_items[0] + "}"
            parts.append(f"selectors={sel_str}")

    if agg:
        parts.append(f"agg={agg}")

    # Format the output
    if len(parts) <= 2 and all(len(p) < 40 for p in parts):
        return "Metric(" + ", ".join(parts) + ")"
    else:
        inner = ",\n    ".join(parts)
        return f"Metric(\n    {inner},\n)"


def extract_dict_pairs(s):
    """Extract key-value pairs from a dict literal string like {"key": value}"""
    result = {}
    # Remove outer braces
    inner = s.strip()
    if inner.startswith("{"):
        inner = inner[1:]
    if inner.endswith("}"):
        inner = inner[:-1]
    inner = inner.strip()
    if not inner:
        return result

    # Split by comma at top level
    parts = split_args(inner)
    for part in parts:
        part = part.strip()
        # Match "key": value or 'key': value
        m = re.match(r"""["'](\w+)["']\s*:\s*(.+)""", part, re.DOTALL)
        if m:
            result[m.group(1)] = m.group(2).strip()
    return result


def process_notebook(fpath):
    """Process a .ipynb file."""
    with open(fpath, "r") as f:
        nb = json.load(f)

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source_lines = cell.get("source", [])
        source_str = "".join(source_lines)

        if "Feedback(" not in source_str:
            continue

        new_source = migrate_source(source_str)
        if new_source != source_str:
            # Convert back to list of lines for notebook format
            new_lines = []
            for line in new_source.split("\n"):
                new_lines.append(line + "\n")
            # Last line shouldn't have trailing newline if original didn't
            if (
                new_lines
                and source_lines
                and not source_lines[-1].endswith("\n")
            ):
                new_lines[-1] = new_lines[-1].rstrip("\n")
            cell["source"] = new_lines
            changed = True

    if changed:
        with open(fpath, "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")

    return changed


def process_py_file(fpath):
    """Process a .py file."""
    with open(fpath, "r") as f:
        source = f.read()

    if "Feedback(" not in source:
        return False

    new_source = migrate_source(source)
    if new_source != source:
        with open(fpath, "w") as f:
            f.write(new_source)
        return True
    return False


def main():
    files = find_files_with_feedback(COOKBOOK_DIR)
    print(f"Found {len(files)} files with Feedback(")

    changed = []
    failed = []
    skipped = []

    for fpath in files:
        rel = os.path.relpath(fpath, os.path.dirname(__file__))
        try:
            if fpath.endswith(".ipynb"):
                if process_notebook(fpath):
                    changed.append(rel)
                    print(f"  MIGRATED: {rel}")
                else:
                    skipped.append(rel)
                    print(f"  SKIPPED (no changes): {rel}")
            elif fpath.endswith(".py"):
                if process_py_file(fpath):
                    changed.append(rel)
                    print(f"  MIGRATED: {rel}")
                else:
                    skipped.append(rel)
                    print(f"  SKIPPED (no changes): {rel}")
        except Exception as e:
            failed.append((rel, str(e)))
            print(f"  FAILED: {rel} - {e}")

    print(f"\n{'=' * 60}")
    print(f"Changed: {len(changed)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed files:")
        for f, e in failed:
            print(f"  {f}: {e}")

    return changed, skipped, failed


if __name__ == "__main__":
    changed, skipped, failed = main()
    sys.exit(1 if failed else 0)
