import numpy as np

from trulens_eval.util import SerialModel


class CATEGORY:
    """
    Feedback result categories for displaying purposes: pass, warning, fail, or
    unknown.
    """

    class Category(SerialModel):
        name: str
        adjective: str
        threshold: float
        color: str
        icon: str

    PASS = Category(
        name="pass", adjective="high", threshold=0.8, color="#aaffaa", icon="✅"
    )
    WARNING = Category(
        name="warning",
        adjective="medium",
        threshold=0.6,
        color="#ffffaa",
        icon="⚠️"
    )
    FAIL = Category(
        name="fail", adjective="low", threshold=0.0, color="#ffaaaa", icon="🛑"
    )
    UNKNOWN = Category(
        name="unknown",
        adjective="unknown",
        threshold=np.nan,
        color="#aaaaaa",
        icon="?"
    )

    ALL = [PASS, WARNING, FAIL]  # not including UNKNOWN intentionally

    @staticmethod
    def of_score(score: float) -> Category:
        for cat in CATEGORY.ALL:
            if score >= cat.threshold:
                return cat

        return CATEGORY.UNKNOWN


# These would be useful to include in our pages but don't yet see a way to do
# this in streamlit.
root_js = f"""
    var default_pass_threshold = {CATEGORY.PASS.threshold};
    var default_warning_threshold = {CATEGORY.WARNING.threshold};
    var default_fail_threshold = {CATEGORY.FAIL.threshold};
"""

# Not presently used. Need to figure out how to include this in streamlit pages.
root_html = f"""
<script>
    {root_js}
</script>
"""

stmetricdelta_hidearrow = """
    <style> [data-testid="stMetricDelta"] svg { display: none; } </style>
    """

cellstyle_jscode = f"""
    function(params) {{
        let v = parseFloat(params.value);
        """ + \
    "\n".join(f"""
        if (v >= {cat.threshold}) {{
            return {{
                'color': 'black',
                'backgroundColor': '{cat.color}'
            }};
        }}
    """ for cat in CATEGORY.ALL) + f"""
        // i.e. not a number
        return {{
            'color': 'black',
            'backgroundColor': '{CATEGORY.UNKNOWN.color}'
        }};
    }}
    """

hide_table_row_index = """
    <style>
        thead tr th:first-child {display:none}
        tbody th {display:none}
    </style>
    """
