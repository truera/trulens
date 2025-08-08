# TruLens Dashboard Custom Tabs

This document explains how to add custom tabs to the TruLens dashboard without modifying the source code.

## Quick Start

1. **Create a Python file** with your custom tab code
2. **Use the registry system** to register your tab
3. **Place the file** in one of the supported locations
4. **Restart the dashboard** to see your new tab

## Example Custom Tab

Here's a minimal example of a custom tab:

```python
import streamlit as st
from trulens.dashboard.registry import register_page

def render_my_tab(app_name: str):
    st.title("My Custom Tab")
    if app_name:
        st.write(f"Analyzing app: {app_name}")
        # Your custom analysis here
    else:
        st.info("Please select an app from the sidebar.")

# Register the tab
register_page("My Custom Tab", render_my_tab)
```

## Where to Place Custom Tab Files

You can place your custom tab files in any of these locations:

### 1. Current Working Directory (Recommended)
Create a `trulens_pages/` directory in your current working directory:
```
your_project/
├── trulens_pages/
│   ├── my_custom_tab.py
│   └── another_tab.py
└── your_main_script.py
```

### 2. Built-in Plugins (For Contributors)
Place files directly in `src/dashboard/trulens/dashboard/plugins/` and update the `__init__.py` file.

## Tab Function Requirements

Your tab render function must:

1. **Accept an `app_name` parameter** (can be `None` if no app is selected)
2. **Use Streamlit components** to render your UI
3. **Be registered** using `register_page(tab_name, render_function)`

```python
def render_my_tab(app_name: str):
    # app_name can be None if no app is selected
    if app_name is None:
        st.info("Please select an app from the sidebar")
        return

    # Your tab content here
    st.title("My Custom Analysis")
    # ... rest of your code
```

## Accessing TruLens Data

You can access TruLens data using the dashboard utilities:

```python
from trulens.dashboard.utils.dashboard_utils import get_records_and_feedback
from trulens.dashboard.utils.dashboard_utils import render_app_version_filters

def render_my_tab(app_name: str):
    # Get app versions for filtering
    versions_df, _ = render_app_version_filters(app_name, {}, page_name_keys=[])
    app_ids = versions_df["app_id"].tolist()

    # Get records and feedback data
    records_df, feedback_col_names = get_records_and_feedback(app_ids=app_ids)

    # Your analysis using the data
    st.dataframe(records_df)
```

## Advanced Features

### Using the Decorator Syntax

You can also use a decorator to register your tabs:

```python
from trulens.dashboard.registry import page

@page("My Decorated Tab")
def render_decorated_tab(app_name: str):
    st.write("This tab was registered with a decorator!")
```

### Conditional Tab Registration

You can conditionally register tabs based on environment variables or other conditions:

```python
import os
from trulens.dashboard.registry import register_page

def render_debug_tab(app_name: str):
    st.title("Debug Information")
    # Debug content here

# Only register the debug tab in development
if os.getenv("TRULENS_DEBUG") == "1":
    register_page("Debug", render_debug_tab)
```

### Page State Management

For tabs that need to maintain state, use Streamlit's session state:

```python
def render_stateful_tab(app_name: str):
    # Initialize state
    if "my_tab_counter" not in st.session_state:
        st.session_state.my_tab_counter = 0

    # Use state
    if st.button("Increment"):
        st.session_state.my_tab_counter += 1

    st.write(f"Counter: {st.session_state.my_tab_counter}")
```

## Template File

See `example_custom_tab.py.template` in this directory for a complete example with charts, metrics, and controls.

## Troubleshooting

**Tab doesn't appear:**
- Check that your file is in one of the supported locations
- Ensure your Python file doesn't start with underscore (`_`)
- Verify you called `register_page()` at the module level
- Restart the dashboard after adding new files

**Import errors:**
- Make sure all required packages are installed
- Check that your custom modules don't conflict with existing names
- Use try/except blocks for optional dependencies

**Tab appears but doesn't work:**
- Check the Streamlit console for error messages
- Ensure your render function accepts the `app_name` parameter
- Test your function independently before registering it

## Built-in Plugin Examples

The `stability.py` plugin in this directory shows a real-world example of a custom tab that:
- Analyzes data quality and consistency
- Uses environment variables for conditional activation
- Implements proper error handling and user feedback
- Follows TruLens dashboard patterns and styling
