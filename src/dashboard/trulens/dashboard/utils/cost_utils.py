from typing import Dict, List, Tuple

import pandas as pd
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.dashboard.utils import dashboard_utils


def process_costs(records_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Process costs in a dataframe to handle both OTEL and non-OTEL cases.

    Args:
        records_df: DataFrame containing cost information

    Returns:
        Tuple of:
        - DataFrame with processed cost columns
        - List of cost column names
    """
    # Handle OTEL case
    if is_otel_tracing_enabled():
        connector = dashboard_utils.get_session().connector
        if connector is not None:
            records_df, cost_cols = connector._create_currency_cost_columns(
                records_df
            )
            # The cost_cols returned by _create_currency_cost_columns already has the "cost_" prefix
            # We just need to create the total_cost columns
            for currency in cost_cols:
                records_df[f"total_cost_{currency.replace('cost_', '')}"] = (
                    records_df[currency]
                )
            return records_df, [col.replace("cost_", "") for col in cost_cols]

    # Handle non-OTEL case or fallback
    records_df["total_cost_usd"] = records_df["total_cost"].where(
        records_df["cost_currency"] == "USD",
        other=0,
    )
    records_df["total_cost_sf"] = records_df["total_cost"].where(
        records_df["cost_currency"] == "Snowflake credits",
        other=0,
    )
    return records_df, ["usd", "sf"]


def get_cost_aggregations(cost_cols: List[str]) -> Dict[str, Tuple[str, str]]:
    """Get cost aggregation dictionary for use in groupby operations.

    Args:
        cost_cols: List of cost column names (without 'cost_' prefix)

    Returns:
        Dictionary mapping cost column names to their aggregation tuples
    """
    return {
        f"Total Cost ({currency.upper()})": (f"total_cost_{currency}", "sum")
        for currency in cost_cols
    }


def format_cost(cost: float, currency: str) -> str:
    """Format a cost value with its currency for display.

    Args:
        cost: The cost value to format
        currency: The currency code (e.g. 'USD', 'sf')

    Returns:
        Formatted cost string
    """
    if currency.lower() == "sf":
        return f"{cost:.5f} Snowflake credits"
    elif currency.lower() == "usd":
        return f"${cost:.2f}"
    else:
        return f"{cost:.2f} {currency.upper()}"
