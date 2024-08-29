# ruff: noqa: E402
"""Creates grit patterns to de-alias imports in trulens_eval and submodules."""

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
from types import ModuleType
from typing import Iterable, Optional, Tuple, Union

# Add base dir to path to be able to access test folder.
base_dir = Path().cwd().parent.parent.resolve()
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

from tests.utils import get_submodule_names


def get_module_source(mod: Union[ModuleType, Path, str]) -> str:
    """Get the source code of the given module, given by a module object, path,
    or qualifed name."""

    if isinstance(mod, ModuleType):
        return inspect.getsource(mod)
    elif isinstance(mod, Path):
        with open(mod, "r") as f:
            return f.read()
    elif isinstance(mod, str):
        loc = importlib.util.find_spec(mod).origin
        if loc is None:
            raise ValueError(
                f"Could not find module {mod}. Might be a namespace module."
            )

        with open(loc, "r") as f:
            return f.read()

    else:
        raise ValueError(f"Unknown type {type(mod).__name__}")


def get_post_dep_imports(mod: Union[ModuleType, Path, str]):
    """Get all of the imports after the deprecation warning call."""

    src = get_module_source(mod)

    mod_ast = ast.parse(src)

    names = {}

    # get ast after the deprecation warning
    statements = list(reversed(mod_ast.body))
    statement = statements.pop()
    while len(statements) > 0:
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "packages_dep_warn"
            and statement.value.func.value.id == "deprecation_utils"
        ):
            break
        statement = statements.pop()

    mod_ast.body = list(reversed(statements))

    relevant = [
        node
        for node in ast.walk(mod_ast)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    for node in relevant:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names[alias.asname or alias.name] = node
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names[alias.asname or alias.name] = node

    return names


def get_import_target(node: ast.AST) -> Optional[Tuple[str, str]]:
    """Return the module and name referred to by the given ast node."""

    if isinstance(node, ast.Import):
        return (node.names[0].asname or node.names[0].name, node.names[0].name)
    elif isinstance(node, ast.ImportFrom):
        return (node.module, node.names[0].name)

    return None


def grit_patterns_dealias(mod: Union[ModuleType, str]) -> Iterable[str]:
    """Generate grit patterns for migrating from imports from the given module
    to what from where this module imports instead."""

    if isinstance(mod, str):
        mod_name = mod
    elif isinstance(mod, ModuleType):
        mod_name = mod.__name__
    else:
        raise ValueError(f"Unknown type {type(mod).__name__}")

    for name, ast_ in get_post_dep_imports(mod).items():
        if (new_qual := get_import_target(ast_)) is None:
            continue

        new_mod, new_name = new_qual

        yield f"`from {mod_name} import {name}` => `from {new_mod} import {new_name}`"
        yield f"`from {mod_name} import {name} as $alias` => `from {new_mod} import {new_name} as $alias`"


if __name__ == "__main__":
    print("""
engine marzano(0.1)
language python

any {
    """)
    for mod_name in get_submodule_names("trulens_eval"):
        for pattern in grit_patterns_dealias(mod_name):
            print("  " + pattern + ",")

print("}")
