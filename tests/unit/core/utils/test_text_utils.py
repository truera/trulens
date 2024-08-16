from typing import Union

import pytest
from trulens.core.utils.text import format_quantity
from trulens.core.utils.text import format_seconds


@pytest.mark.parametrize(
    "value,precision,expected",
    [
        (-7654, 2, "-7.65k"),
        (0, 0, "0"),
        (0, 1, "0.0"),
        (0, 2, "0.00"),
        (1, 0, "1"),
        (1e2, 0, "100"),
        (1200, 1, "1.2k"),
        (1250, 1, "1.2k"),
        (1251, 1, "1.3k"),
        (3450000, 2, "3.45M"),
        (3458000, 2, "3.46M"),
        (1e8, 0, "100M"),
        (1e10, 2, "10.00B"),
    ],
)
def test_format_quantity(
    value: Union[int, float], precision: int, expected: str
):
    assert format_quantity(value, precision=precision) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "0 seconds"),
        (1, "1 second"),
        (60, "1 minute"),
        (61, "1 minute 1 second"),
        (3599, "59 minutes 59 seconds"),
        (3600, "1 hour"),
        (3662, "1 hour 1 minute"),
        (3600 * 6, "6 hours"),
        (86400, "1 day"),
        ((86400 * 2) + (60 * 60), "2 days 1 hour"),
        ((86400 * 2) + (2 * 60 * 60), "2 days 2 hours"),
    ],
)
def test_format_seconds(value: int, expected: str):
    assert format_seconds(value) == expected
