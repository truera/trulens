import json

import pandas
import pytest
from trulens.core import session as core_session


class _FakeConnector:
    def __init__(self, leaderboard: pandas.DataFrame):
        self.leaderboard = leaderboard
        self.calls = []

    def get_leaderboard(self, **kwargs):
        self.calls.append(kwargs)
        return self.leaderboard


@pytest.fixture
def leaderboard_session():
    leaderboard = pandas.DataFrame({
        "app_name": ["app-a", "app-b"],
        "latency": [0.1, 0.2],
        "quality": [0.9, 0.8],
    })
    connector = _FakeConnector(leaderboard)
    session = core_session.TruSession.model_construct(connector=connector)
    return session, connector, leaderboard


def test_get_leaderboard_defaults_to_dataframe(leaderboard_session):
    session, connector, leaderboard = leaderboard_session

    result = session.get_leaderboard(app_ids=["app-a"], limit=1, offset=2)

    assert result is leaderboard
    assert connector.calls == [
        {
            "app_ids": ["app-a"],
            "group_by_metadata_key": None,
            "limit": 1,
            "offset": 2,
        }
    ]


def test_get_leaderboard_returns_dict_records(leaderboard_session):
    session, _, _ = leaderboard_session

    result = session.get_leaderboard(format="dict")

    assert result == [
        {"app_name": "app-a", "latency": 0.1, "quality": 0.9},
        {"app_name": "app-b", "latency": 0.2, "quality": 0.8},
    ]


def test_get_leaderboard_returns_json_records(leaderboard_session):
    session, _, _ = leaderboard_session

    result = session.get_leaderboard(format="json")

    assert json.loads(result) == [
        {"app_name": "app-a", "latency": 0.1, "quality": 0.9},
        {"app_name": "app-b", "latency": 0.2, "quality": 0.8},
    ]


def test_get_leaderboard_rejects_unknown_format(leaderboard_session):
    session, _, _ = leaderboard_session

    with pytest.raises(ValueError, match="Unsupported leaderboard format"):
        session.get_leaderboard(format="yaml")
