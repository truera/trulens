from enum import Enum
import operator
from typing import Callable, List, NamedTuple, Optional

import numpy as np
from trulens.core.utils.serial import SerialModel


class ResultCategoryType(Enum):
    PASS = 0
    WARNING = 1
    FAIL = 2


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


class CATEGORY:
    """
    Feedback result categories for displaying purposes: pass, warning, fail, or
    unknown.
    """

    # support both directions by default
    # TODO: make this configurable (per feedback definition & per app?)
    directions = [
        FeedbackDirection("HIGHER_IS_BETTER", False, [0, 0.6, 0.8]),
        FeedbackDirection("LOWER_IS_BETTER", True, [0.2, 0.4, 1]),
    ]

    styling = {
        "PASS": dict(color="#aaffaa44", icon="✅"),
        "WARNING": dict(color="#ffffaa44", icon="⚠️"),
        "FAIL": dict(color="#ffaaaa44", icon="🛑"),
    }

    PASS: dict[str, Category] = {}
    FAIL: dict[str, Category] = {}
    WARNING: dict[str, Category] = {}

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
                if direction.name == "HIGHER_IS_BETTER"
                else operator.le,
                **styling[category_name],
            )

    UNKNOWN = Category(
        name="unknown",
        adjective="unknown",
        threshold=np.nan,
        color="#aaaaaa44",
        icon="?",
    )

    # order matters here because `of_score` returns the first best category
    ALL = [PASS, WARNING, FAIL]  # not including UNKNOWN intentionally

    @staticmethod
    def of_score(score: float, higher_is_better: bool = True) -> Category:
        direction_key = (
            "HIGHER_IS_BETTER" if higher_is_better else "LOWER_IS_BETTER"
        )

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

cell_rules_styles = {
    f".cat-{cat_name.lower()}": {"background-color": styles["color"]}
    for cat_name, styles in CATEGORY.styling.items()
}

aggrid_css = {
    **cell_rules_styles,
    ".ag-row .ag-cell": {
        "display": "flex",
        "align-items": "center",
    },
}

cell_rules = {}
for direction in valid_directions:
    categories: List[Category] = [
        CATEGORY.FAIL[direction],
        CATEGORY.WARNING[direction],
        CATEGORY.PASS[direction],
    ]
    thresholds = [cat.threshold for cat in categories]
    direction_rules = {}
    direction_rules[f"cat-{CATEGORY.UNKNOWN.name.lower()}"] = "x == null"
    for i, cat in enumerate(categories):
        op = ">=" if cat.compare is operator.ge else "<"
        upper_op = "<" if cat.compare is operator.ge else ">="

        css_class = f"cat-{cat.name}"
        if i < len(categories) - 1:
            direction_rules[css_class] = (
                f"x {op} {cat.threshold} && x {upper_op} {categories[i + 1].threshold}"
            )
        else:
            direction_rules[css_class] = f"x {op} {cat.threshold}"
    cell_rules[direction] = direction_rules

cell_rules_styles[f".cat-{CATEGORY.UNKNOWN.name.lower()}"] = {
    "background-color": CATEGORY.UNKNOWN.color
}

cellstyle_jscode = {
    k: """function(params) {
        let v = parseFloat(params.value);
        """
    + "\n".join(
        f"""
        if (v {">=" if k == "HIGHER_IS_BETTER" else "<="} {cat.threshold}) {{
            return {{
                'backgroundColor': '{cat.color}'
            }};
        }}
    """
        for cat in map(operator.itemgetter(k), CATEGORY.ALL)
    )
    + f"""
        // i.e. not a number
        return {{
            'backgroundColor': '{CATEGORY.UNKNOWN.color}'
        }};
    }}"""
    for k in valid_directions
}

hide_table_row_index = """
    <style>
        thead tr th:first-child {display:none}
        tbody th {display:none}
    </style>
    """
