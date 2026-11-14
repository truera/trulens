import pytest
from trulens.core.utils import json as json_utils
from trulens.core.utils import keys as key_utils


@pytest.mark.parametrize(
    "key",
    [
        "password",
        "Password",
        "db_password",
        "SNOWFLAKE_PASSWORD",
        "api_key",
        "openai_api_key",
        "OPENAI_API_KEY",
        "API_KEY",
        "Api_Key",
        "OpenAI_API_Key",
        "hf_api_token",
        "HF_API_TOKEN",
        "KAGGLE_KEY",
        "kaggle_key",
        "SLACK_TOKEN",
        "slack_token",
        "slack_signing_secret",
    ],
)
def test_should_redact_key(key: str):
    assert key_utils.should_redact_key(key)


@pytest.mark.parametrize(
    "key", ["user", "account", "max_tokens", "token", "kaggle_user", None]
)
def test_should_not_redact_key(key):
    assert not key_utils.should_redact_key(key)


def test_jsonify_redacts_lowercase_password():
    connection_parameters = {
        "account": "my_account",
        "user": "my_user",
        "password": "hunter2",
    }

    redacted = json_utils.jsonify(connection_parameters, redact_keys=True)

    assert redacted == {
        "account": "my_account",
        "user": "my_user",
        "password": key_utils.REDACTED_VALUE,
    }
