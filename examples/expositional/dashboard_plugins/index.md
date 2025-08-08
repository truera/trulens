# TruLens Dashboard Custom Tabs System

This guide explains the new extensible tab system for the TruLens dashboard that allows you to add custom tabs without modifying the source code.

## Overview

The TruLens dashboard now supports custom tabs through a simple registration system. You can create custom analysis pages, visualizations, and tools that integrate seamlessly with the existing dashboard.

## Quick Start

### 1. Create Your Custom Tab

Create a Python file with your custom tab code:

```python
# my_custom_tab.py
import streamlit as st
from trulens.dashboard.registry import register_page

def render_my_analysis(app_name: str):
    st.title("My Custom Analysis")
    if app_name:
        st.write(f"Analyzing app: {app_name}")
        # Your custom analysis here
    else:
        st.info("Please select an app from the sidebar")

# Register the tab
register_page("My Analysis", render_my_analysis)
```

### 2. Place the File

Put your file in the `trulens_pages/` directory in your working directory.

### 3. Restart the Dashboard

Run your TruLens dashboard as usual. Your custom tab will appear automatically!

## File Locations

### Working Directory (Recommended)

Create a `trulens_pages/` directory in your current working directory:

```
your_project/
├── trulens_pages/          # Create this directory
│   ├── my_analysis.py      # Your custom tabs
│   ├── quality_metrics.py
│   └── custom_reports.py
├── your_app.py
└── run_dashboard.py
```

## Tab Function Requirements

Your tab render function must:

1. **Accept a `app_name` parameter that's a `str` or `None`**
2. **Use Streamlit components** for the UI
3. **Be registered** using `register_page()`

```python
def render_my_tab(app_name: str):
    # Handle case where no app is selected
    if app_name is None:
        st.info("Please select an app from the sidebar")
        return

    # Your tab content here
    st.title("My Custom Tab")
    # ... rest of your code
```

## Accessing TruLens Data

Use the dashboard utilities to access TruLens data:

```python
from trulens.dashboard.utils.dashboard_utils import get_records_and_feedback
from trulens.dashboard.utils.dashboard_utils import render_app_version_filters

def render_data_analysis_tab(app_name: str):
    # Get app versions and IDs
    versions_df, _ = render_app_version_filters(
        app_name, {}, page_name_keys=[]
    )
    app_ids = versions_df["app_id"].tolist()

    # Get records and feedback data
    records_df, feedback_col_names = get_records_and_feedback(
        app_ids=app_ids
    )

    # Analyze the data
    st.dataframe(records_df)
    st.write(f"Available feedback columns: {feedback_col_names}")
```

## Examples

### Simple Metrics Tab

```python
import streamlit as st
from trulens.dashboard.registry import register_page

def render_metrics_tab(app_name: str):
    st.title("📊 Custom Metrics")

    if not app_name:
        st.info("Select an app to view metrics")
        return

    # Display custom metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Success Rate", "94.2%", "2.1%")
    with col2:
        st.metric("Avg Response Time", "1.3s", "-0.2s")
    with col3:
        st.metric("Total Requests", "1,247", "156")

register_page("Metrics", render_metrics_tab)
```

### Data Visualization Tab

```python
import streamlit as st
import pandas as pd
import numpy as np
from trulens.dashboard.registry import register_page

def render_charts_tab(app_name: str):
    st.title("📈 Custom Charts")

    if not app_name:
        st.info("Select an app to view charts")
        return

    # Generate sample data
    data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=30),
        'accuracy': np.random.normal(0.85, 0.05, 30)
    })

    # Display chart
    st.line_chart(data.set_index('date'))

register_page("Charts", render_charts_tab)
```

### Conditional Registration

```python
import os
from trulens.dashboard.registry import register_page

def render_debug_tab(app_name: str):
    st.title("🔧 Debug Information")
    # Debug content here

# Only show debug tab in development
if os.getenv("TRULENS_DEBUG") == "1":
    register_page("Debug", render_debug_tab)
```

## Advanced Features

### Using Decorators

```python
from trulens.dashboard.registry import page

@page("My Decorated Tab")
def render_decorated_tab(app_name: str):
    st.write("This tab was registered with a decorator!")
```

### State Management

```python
def render_stateful_tab(app_name: str):
    # Initialize state
    if "counter" not in st.session_state:
        st.session_state.counter = 0

    # Use state
    if st.button("Increment"):
        st.session_state.counter += 1

    st.write(f"Counter: {st.session_state.counter}")
```

### Error Handling

```python
def render_robust_tab(app_name: str):
    try:
        # Your analysis code
        result = complex_analysis(app_name)
        st.success("Analysis completed successfully!")
        st.write(result)
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.info("Please check your data and try again.")
```

## Troubleshooting

**Tab doesn't appear:**

- Verify the file is in the `trulens_pages/` directory
- Check that the filename doesn't start with underscore
- Ensure `register_page()` is called at the module level
- Restart the dashboard after adding new files

**Import errors:**

- Install required packages
- Check for naming conflicts
- Use try/except for optional dependencies

**Tab appears but crashes:**

- Check Streamlit console for errors
- Ensure function accepts `app_name` parameter
- Test the function independently first

## Best Practices

1. **Use descriptive tab names** that clearly indicate the tab's purpose
2. **Handle the `app_name=None` case** gracefully
3. **Provide user feedback** for loading states and errors
4. **Use Streamlit's caching** for expensive computations
5. **Follow TruLens styling patterns** for consistency
6. **Test your tabs independently** before integrating
7. **Document your custom tabs** for team members

## Support

For questions or issues with custom tabs:

- Check the examples in this repository
- Review the built-in plugin implementations
- Look at the template file for guidance
- Submit issues to the TruLens GitHub repository
