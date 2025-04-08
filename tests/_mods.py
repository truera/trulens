# ruff: noqa: E402, F401

"""Import renames for TruLens test modules as per code standards.

This is intentionally a Python file to be analyzed by static tools. Do not
import this module otherwise.
"""

from tests import test as mod_test  # not a good name
from tests import utils as test_utils
from tests.unit import feedbacks as test_feedbacks

# Modules that should not be imported at all:
"""
./docs_notebooks/test_notebooks.py
./unit/test_feedback.py
./unit/test_tru_basic_app.py
./unit/core/utils/test_json_utils.py
./unit/core/utils/test_text_utils.py
./unit/test_lens.py
./unit/test_feedback_score_generation.py
./unit/static/test_deprecation.py
./unit/static/test_static.py
./unit/static/test_api.py
./unit/test_app.py
./unit/test_tru_custom.py
./util/snowflake_test_case.py
./integration/test_database.py
./legacy/test_trulens_eval_notebooks.py
./e2e/test_serial.py
./e2e/test_providers.py
./e2e/test_tru_chain.py
./e2e/test_endpoints.py
./e2e/test_dummy.py
./e2e/test_tru_llama.py
./e2e/test_embedding_feedback.py
./e2e/test_tru_session.py
./e2e/test_snowflake_connection.py
./e2e/test_snowflake_feedback_evaluation.py
"""
