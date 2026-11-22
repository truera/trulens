"""Tests for the coding-agent columns on the Records tab (steps 5-6 of
truera/trulens#2879): `verification` and `test_file_edits` are shown and
filterable when present in the records dataframe.
"""

import pandas as pd
from trulens.dashboard.tabs.Records import _build_grid_options


def _records_df():
    return pd.DataFrame({
        "app_version": ["v1"],
        "app_name": ["coding-agent"],
        "app_id": ["app-1"],
        "record_id": ["record-1"],
        "verification": ["passed"],
        "test_file_edits": [2],
    })


def test_records_grid_shows_verification_columns_filterable():
    grid_options = _build_grid_options(
        _records_df(),
        feedback_col_names=[],
        feedback_directions={},
        version_metadata_col_names=[],
    )
    column_defs = {col["field"]: col for col in grid_options["columnDefs"]}

    assert column_defs["verification"]["headerName"] == "Verification"
    assert column_defs["verification"]["filter"] == "agSetColumnFilter"
    assert column_defs["test_file_edits"]["headerName"] == "Test File Edits (#)"
    assert column_defs["test_file_edits"]["filter"] == "agNumberColumnFilter"
