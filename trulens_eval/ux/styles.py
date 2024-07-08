from collections import defaultdict
from enum import Enum
import operator
from typing import Callable, List, NamedTuple, Optional

import numpy as np

from trulens_eval.utils.serial import SerialModel


class ResultCategoryType(Enum):
    PASS = 0
    WARNING = 1
    FAIL = 2


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
        direction: Optional[str] = None
        compare: Optional[Callable[[float, float], bool]] = None

    class FeedbackDirection(NamedTuple):
        name: str
        ascending: bool
        thresholds: List[float]

    # support both directions by default
    # TODO: make this configurable (per feedback definition & per app?)
    directions = [
        FeedbackDirection("HIGHER_IS_BETTER", False, [0, 0.6, 0.8]),
        FeedbackDirection("LOWER_IS_BETTER", True, [0.2, 0.4, 1]),
    ]

    styling = {
        "PASS": dict(color="#aaffaa", icon="✅"),
        "WARNING": dict(color="#ffffaa", icon="⚠️"),
        "FAIL": dict(color="#ffaaaa", icon="🛑"),
    }

    for category_name in ResultCategoryType._member_names_:
        locals()[category_name] = defaultdict(dict)

    for direction in directions:
        a = sorted(
            zip(["low", "medium", "high"], sorted(direction.thresholds)),
            key=operator.itemgetter(1),
            reverse=not direction.ascending,
        )

        for enum, (adjective, threshold) in enumerate(a):
            category_name = ResultCategoryType(enum).name
            locals()[category_name][direction.name] = Category(
                name=category_name.lower(),
                adjective=adjective,
                threshold=threshold,
                direction=direction.name,
                compare=operator.ge
                if direction.name == "HIGHER_IS_BETTER" else operator.le,
                **styling[category_name],
            )

    UNKNOWN = Category(
        name="unknown",
        adjective="unknown",
        threshold=np.nan,
        color="#aaaaaa",
        icon="?"
    )

    # order matters here because `of_score` returns the first best category
    ALL = [PASS, WARNING, FAIL]  # not including UNKNOWN intentionally

    @staticmethod
    def of_score(score: float, higher_is_better: bool = True) -> Category:
        direction_key = "HIGHER_IS_BETTER" if higher_is_better else "LOWER_IS_BETTER"

        for cat in map(operator.itemgetter(direction_key), CATEGORY.ALL):
            if cat.compare(score, cat.threshold):
                return cat

        return CATEGORY.UNKNOWN


default_direction = "HIGHER_IS_BETTER"

# These would be useful to include in our pages but don't yet see a way to do
# this in streamlit.
root_js = f"""
    var default_pass_threshold = {CATEGORY.PASS[default_direction].threshold};
    var default_warning_threshold = {CATEGORY.WARNING[default_direction].threshold};
    var default_fail_threshold = {CATEGORY.FAIL[default_direction].threshold};
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

valid_directions = ["HIGHER_IS_BETTER", "LOWER_IS_BETTER"]

cellstyle_jscode = {
    k: f"""function(params) {{
        let v = parseFloat(params.value);
        """ + "\n".join(
        f"""
        if (v {'>=' if k == "HIGHER_IS_BETTER" else '<='} {cat.threshold}) {{
            return {{
                'color': 'black',
                'backgroundColor': '{cat.color}'
            }};
        }}
    """ for cat in map(operator.itemgetter(k), CATEGORY.ALL)
    ) + f"""
        // i.e. not a number
        return {{
            'color': 'black',
            'backgroundColor': '{CATEGORY.UNKNOWN.color}'
        }};
    }}""" for k in valid_directions
}

hide_table_row_index = """
    <style>
        thead tr th:first-child {display:none}
        tbody th {display:none}
    </style>
    """
