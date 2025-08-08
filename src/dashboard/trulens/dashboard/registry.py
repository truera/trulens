"""Registry system for custom dashboard pages."""

import importlib
from pathlib import Path
import sys
from typing import Callable, Dict

import streamlit as st

# Global registry for custom pages.
_registered_pages: Dict[str, Callable] = {}


def register_page(name: str, render_function: Callable) -> None:
    """Register a custom page with the dashboard.

    Args:
        name: The name of the page (will appear as tab name)
        render_function:
            Function that renders the page content, should accept `app_name`
            parameter of type `typing.Optional[str]`.
    """
    _registered_pages[name] = render_function


def get_registered_pages() -> Dict[str, Callable]:
    """Get all registered custom pages."""
    return _registered_pages.copy()


def load_custom_pages() -> None:
    """
    Load custom pages from trulens_pages directory in current working directory.
    """
    cwd_pages_dir = Path.cwd() / "trulens_pages"
    if cwd_pages_dir.exists() and cwd_pages_dir.is_dir():
        _load_from_directory(cwd_pages_dir)


def _load_from_directory(directory: Path) -> None:
    """Load all Python modules from a directory."""
    if not directory.exists() or not directory.is_dir():
        return
    # Add directory to Python path temporarily
    str_dir = str(directory)
    if str_dir not in sys.path:
        sys.path.insert(0, str_dir)

    try:
        # Load all .py files in the directory
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_"):
                continue  # Skip private modules

            module_name = py_file.stem
            try:
                if module_name in sys.modules:
                    # Reload if already imported
                    importlib.reload(sys.modules[module_name])
                else:
                    importlib.import_module(module_name)
            except Exception as e:
                st.warning(
                    f"Failed to load custom page module {module_name}: {e}"
                )

    finally:
        # Clean up path
        if str_dir in sys.path:
            sys.path.remove(str_dir)


def create_page_decorator(name: str):
    """Create a decorator for registering pages.

    Usage:
        @create_page_decorator("My Custom Tab")
        def render_my_tab(app_name: str):
            st.write(f"Hello from {app_name}")
    """

    def decorator(func: Callable):
        register_page(name, func)
        return func

    return decorator


# Convenience decorator.
page = create_page_decorator
