"""
Example of how to create a custom tab for TruLens Dashboard.

This example shows how to:
1. Create a simple custom tab
2. Place it in the correct location
3. Test it with the dashboard

To use this example:
1. Create a directory called 'trulens_pages' in your working directory
2. Copy this file to 'trulens_pages/my_custom_tab.py'
3. Run the TruLens dashboard
4. You should see a "My Custom Tab" appear in the tabs

You can also run this file directly to test the tab function.
"""

import numpy as np
import pandas as pd
import streamlit as st

# Import the registry system
from trulens.dashboard.registry import register_page


def render_custom_analysis_tab(app_name: str):
    """Render a custom analysis tab.

    Args:
        app_name: The currently selected app name, or None if no app selected.
    """
    st.title("🔍 Custom Analysis Tab")

    if app_name is None:
        st.info(
            "👈 Please select an app from the sidebar to see custom analysis"
        )
        st.markdown("---")
        st.markdown("### About This Custom Tab")
        st.markdown(
            "This is an example custom tab that demonstrates how to extend "
            "the TruLens dashboard with your own analysis and visualizations."
        )
        return

    st.markdown(f"**Analyzing app:** `{app_name}`")

    # Example: Custom metrics
    st.subheader("📊 Custom Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Custom Score",
            value="87.3%",
            delta="2.1%",
            help="This is a custom metric you can calculate",
        )

    with col2:
        st.metric(
            label="Processing Time",
            value="1.2s",
            delta="-0.3s",
            help="Average processing time",
        )

    with col3:
        st.metric(label="Success Rate", value="94.5%", delta="1.8%")

    with col4:
        st.metric(label="Total Queries", value="1,247", delta="156")

    # Example: Custom chart
    st.subheader("📈 Custom Visualization")

    # Generate sample data for demonstration
    days = pd.date_range(start="2024-01-01", periods=30, freq="D")
    performance_data = pd.DataFrame({
        "Date": days,
        "Accuracy": np.random.normal(0.85, 0.05, 30).clip(0.7, 1.0),
        "Response Time": np.random.normal(1.5, 0.3, 30).clip(0.5, 3.0),
        "User Satisfaction": np.random.normal(4.2, 0.4, 30).clip(3.0, 5.0),
    })

    # Create tabs for different visualizations
    chart_tab1, chart_tab2, chart_tab3 = st.tabs([
        "📊 Accuracy Trend",
        "⏱️ Response Time",
        "😊 Satisfaction",
    ])

    with chart_tab1:
        st.line_chart(performance_data.set_index("Date")["Accuracy"])
        st.caption("Daily accuracy scores over the past month")

    with chart_tab2:
        st.area_chart(performance_data.set_index("Date")["Response Time"])
        st.caption("Response time trends (seconds)")

    with chart_tab3:
        st.bar_chart(performance_data.set_index("Date")["User Satisfaction"])
        st.caption("User satisfaction ratings (1-5 scale)")

    # Example: Interactive controls
    st.subheader("🎛️ Custom Controls")

    col1, col2 = st.columns(2)

    with col1:
        analysis_type = st.selectbox(
            "Choose Analysis Type:",
            [
                "Performance Analysis",
                "Error Analysis",
                "Usage Patterns",
                "Quality Metrics",
            ],
            help="Select the type of analysis to perform",
        )

        time_range = st.slider(
            "Time Range (days):",
            min_value=1,
            max_value=30,
            value=7,
            help="Select the number of days to analyze",
        )

    with col2:
        include_weekends = st.checkbox("Include Weekends", value=True)
        show_details = st.checkbox("Show Detailed View", value=False)

        if st.button("🔄 Refresh Analysis"):
            st.success(f"Refreshed {analysis_type} for last {time_range} days")

    # Example: Data table
    if show_details:
        st.subheader("📋 Detailed Data")

        # Show the sample data
        filtered_data = performance_data.tail(time_range)
        if not include_weekends:
            filtered_data = filtered_data[filtered_data["Date"].dt.weekday < 5]

        st.dataframe(filtered_data, use_container_width=True, hide_index=True)

        # Download button
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"{app_name}_custom_analysis.csv",
            mime="text/csv",
        )

    # Example: Custom alerts/insights
    st.subheader("💡 Insights & Recommendations")

    if performance_data["Accuracy"].tail(7).mean() < 0.8:
        st.warning(
            "⚠️ Accuracy has been below 80% in the last week. "
            "Consider reviewing your model."
        )

    if performance_data["Response Time"].tail(3).mean() > 2.0:
        st.error("🚨 Response times are elevated. Check system performance.")

    if performance_data["User Satisfaction"].tail(5).mean() > 4.0:
        st.success("🎉 User satisfaction is high! Great job!")

    st.info(
        "💡 **Tip:** This is completely customizable! You can replace this with your own analysis logic."
    )


# Register the tab with the dashboard
# This is what makes the tab appear in the TruLens dashboard
register_page("My Custom Tab", render_custom_analysis_tab)


# For testing the function directly
if __name__ == "__main__":
    st.set_page_config(
        page_title="Custom Tab Test", page_icon="🔍", layout="wide"
    )

    st.sidebar.markdown("### Test Mode")
    test_app_name = st.sidebar.text_input(
        "App Name (for testing)", value="test_app"
    )

    if test_app_name:
        render_custom_analysis_tab(test_app_name)
    else:
        render_custom_analysis_tab(None)
