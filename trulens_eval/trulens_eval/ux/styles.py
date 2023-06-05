from trulens_eval.tru_feedback import default_pass_fail_color_threshold

stmetricdelta_hidearrow = """
    <style> [data-testid="stMetricDelta"] svg { display: none; } </style>
"""

cellstyle_jscode = f"""
    function(params) {{
        if (parseFloat(params.value) < {default_pass_fail_color_threshold}) {{
            return {{
                'color': 'black',
                'backgroundColor': '#FCE6E6'
            }}
        }} else if (parseFloat(params.value) >= {default_pass_fail_color_threshold}) {{
            return {{
                'color': 'black',
                'backgroundColor': '#4CAF50'
            }}
        }} else {{
            return {{
                'color': 'black',
                'backgroundColor': 'white'
            }}
        }}
    }};
        """

hide_table_row_index = """
    <style>
    thead tr th:first-child {display:none}
    tbody th {display:none}
    </style>
"""
