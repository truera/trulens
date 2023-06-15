from trulens_eval.feedback import default_pass_fail_color_threshold

# These would be useful to include in our pages but don't yet see a way to do this in streamlit.
root_js = f"""
    default_pass_fail_color_threshold = {default_pass_fail_color_threshold};
"""

root_html = f"""
js:
<script>
    {root_js}
</script>
"""

stmetricdelta_hidearrow = """
    <style> [data-testid="stMetricDelta"] svg { display: none; } </style>
    """

cellstyle_jscode = """
    function(params) {
        if (parseFloat(params.value) < """ + str(
    default_pass_fail_color_threshold
) + """) {
            return {
                'color': 'black',
                'backgroundColor': '#FCE6E6'
            }
        } else if (parseFloat(params.value) >= """ + str(
    default_pass_fail_color_threshold
) + """) {
            return {
                'color': 'black',
                'backgroundColor': '#4CAF50'
            }
        } else {
            return {
                'color': 'black',
                'backgroundColor': 'white'
            }
        }
    };
    """

hide_table_row_index = """
    <style>
        thead tr th:first-child {display:none}
        tbody th {display:none}
    </style>
    """
