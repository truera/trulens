# TruLens Dashboard Custom Tabs System

This guide explains how to add custom tabs to the TruLens dashboard using environment variables and streamlit components.

## Overview

The TruLens dashboard supports custom tabs by pointing to a directory containing your custom tab files via an environment variable. You can create custom analysis pages, visualizations, and tools that integrate seamlessly with the existing dashboard.

## Quick Start

### 1. Create Your Custom Tab

Create a Python file with your custom tab code:

```python
# my_custom_tab.py
import streamlit as st

def main():
    st.title("My Custom Analysis")

    # Get the selected app from session state (set by the main dashboard)
    app_name = st.session_state.get('selected_app', None)

    if app_name:
        st.write(f"Analyzing app: {app_name}")
        # Your custom analysis here
    else:
        st.info("Please select an app from the sidebar")

if __name__ == "__main__":
    main()
```

### 2. Set Up Your Directory

Create a directory for your custom tabs:

```
your_project/
├── custom_tabs/               # Your custom tabs directory
│   ├── my_analysis.py         # Your custom tabs
│   ├── quality_metrics.py
│   └── custom_reports.py
├── your_app.py
└── run_dashboard.py
```

### 3. Set Environment Variable and Run Dashboard

Set the environment variable to point to your custom tabs directory and run the dashboard:

```python
import os
from trulens.dashboard import run_dashboard

# Set the environment variable
os.environ["TRULENS_UI_CUSTOM_PAGES"] = "/path/to/your/custom_tabs"

# Run the dashboard
run_dashboard()
```

Or set it in your shell:

```bash
export TRULENS_UI_CUSTOM_PAGES="/path/to/your/custom_tabs"
python run_dashboard.py
```

## Tab File Requirements

Your tab files must:

1. **Have a `main()` function** that contains your streamlit code
2. **Include the `if __name__ == "__main__": main()` guard**
3. **Access the selected app** via `st.session_state.get('selected_app', None)`
4. **Use Streamlit components** for the UI

```python
import streamlit as st

def main():
    # Handle case where no app is selected
    app_name = st.session_state.get('selected_app', None)
    if app_name is None:
        st.info("Please select an app from the sidebar")
        return

    # Your tab content here
    st.title("My Custom Tab")
    # ... rest of your code

if __name__ == "__main__":
    main()
```

## Accessing TruLens Data

Use the dashboard utilities to access TruLens data:

```python
import streamlit as st
from trulens.dashboard.utils.dashboard_utils import get_records_and_feedback
from trulens.dashboard.utils.dashboard_utils import render_app_version_filters

def main():
    app_name = st.session_state.get('selected_app', None)
    if not app_name:
        st.info("Please select an app from the sidebar")
        return

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

if __name__ == "__main__":
    main()
```

## Examples

### Simple Metrics Tab

```python
# metrics.py
import streamlit as st

def main():
    st.title("📊 Custom Metrics")

    app_name = st.session_state.get('selected_app', None)
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

if __name__ == "__main__":
    main()
```

### Data Visualization Tab

```python
# charts.py
import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.title("📈 Custom Charts")

    app_name = st.session_state.get('selected_app', None)
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

if __name__ == "__main__":
    main()
```

### Conditional Tab Loading

```python
# debug.py
import os
import streamlit as st

def main():
    st.title("🔧 Debug Information")

    app_name = st.session_state.get('selected_app', None)
    if not app_name:
        st.info("Select an app to view debug info")
        return

    # Debug content here
    st.json({"app_name": app_name, "debug_mode": True})

# Only load this file if debug mode is enabled
if __name__ == "__main__" and os.getenv("TRULENS_DEBUG") == "1":
    main()
```

## Advanced Features

### State Management

```python
def main():
    # Initialize state
    if "counter" not in st.session_state:
        st.session_state.counter = 0

    # Use state
    if st.button("Increment"):
        st.session_state.counter += 1

    st.write(f"Counter: {st.session_state.counter}")

if __name__ == "__main__":
    main()
```

### Error Handling

```python
def main():
    app_name = st.session_state.get('selected_app', None)
    if not app_name:
        st.info("Select an app to view analysis")
        return

    try:
        # Your analysis code
        result = complex_analysis(app_name)
        st.success("Analysis completed successfully!")
        st.write(result)
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.info("Please check your data and try again.")

if __name__ == "__main__":
    main()
```

## Environment Variable Configuration

### Setting the Custom Pages Directory

You can set the `TRULENS_UI_CUSTOM_PAGES` environment variable in several ways:

**In Python:**
```python
import os
os.environ["TRULENS_UI_CUSTOM_PAGES"] = "/absolute/path/to/custom_tabs"
```

**In Shell:**
```bash
export TRULENS_UI_CUSTOM_PAGES="/absolute/path/to/custom_tabs"
```

**In Docker:**
```dockerfile
ENV TRULENS_UI_CUSTOM_PAGES="/app/custom_tabs"
```

### Multiple Directories

You can specify multiple directories by separating them with colons (Unix) or semicolons (Windows):

```bash
export TRULENS_UI_CUSTOM_PAGES="/path/to/tabs1:/path/to/tabs2"
```

## Troubleshooting

**Tab doesn't appear:**

- Verify the environment variable `TRULENS_UI_CUSTOM_PAGES` is set correctly
- Check that the file has a `main()` function
- Ensure the filename doesn't start with underscore
- Check that the file includes `if __name__ == "__main__": main()`
- Restart the dashboard after adding new files

**Import errors:**

- Install required packages
- Check for naming conflicts
- Use try/except for optional dependencies

**Tab appears but crashes:**

- Check Streamlit console for errors
- Ensure the `main()` function exists and is properly defined
- Test the function independently first
- Check that you're accessing `st.session_state.get('selected_app', None)` correctly

## Best Practices

1. **Use descriptive filenames** that clearly indicate the tab's purpose
2. **Handle the `app_name=None` case** gracefully
3. **Provide user feedback** for loading states and errors
4. **Use Streamlit's caching** for expensive computations
5. **Follow TruLens styling patterns** for consistency
6. **Test your tabs independently** before integrating
7. **Document your custom tabs** for team members
8. **Use absolute paths** for the environment variable when possible

## Support

For questions or issues with custom tabs:

- Check the examples in this repository
- Review the built-in tab implementations
- Submit issues to the TruLens GitHub repository
