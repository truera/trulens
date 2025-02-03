# ruff: noqa: E402, F401
"""Import renames for TruLens modules as per code standards.

This is intentionally a python file to be analyzed by static tools. Do not
import this module otherwise.

"""

from trulens.apps import basic as basic_app
from trulens.apps import custom as custom_app
from trulens.apps import virtual as virtual_app
from trulens.apps.langchain import guardrails as langchain_guardrails
from trulens.apps.langchain import langchain as langchain_app
from trulens.apps.langchain import tru_chain as mod_tru_chain  # not a good name
from trulens.apps.llamaindex import guardrails as llama_guardrails
from trulens.apps.llamaindex import llama as llama_app
from trulens.apps.llamaindex import (
    tru_llama as mod_tru_llama,  # not a good name
)
from trulens.apps.nemo import tru_rails as nemo_app  # not a good name
from trulens.connectors.snowflake import connector as snowflake_connector
from trulens.core import app as core_app
from trulens.core import instruments as core_instruments
from trulens.core import session as core_session
from trulens.core._utils import debug as debug_utils
from trulens.core._utils import optional as optional_utils
from trulens.core._utils import pycompat as pycompat_utils
from trulens.core.database import base as core_db
from trulens.core.database import exceptions as db_exceptions
from trulens.core.database import migrations as db_migrations
from trulens.core.database import orm as db_orm
from trulens.core.database import sqlalchemy as db_sqlalchemy
from trulens.core.database import utils as database_utils
from trulens.core.database.connector import base as core_connector
from trulens.core.database.connector import default as default_connector
from trulens.core.database.legacy import migration as db_legacy_migrations
from trulens.core.database.migrations import data as db_data_migrations
from trulens.core.feedback import endpoint as core_endpoint
from trulens.core.feedback import feedback as core_feedback
from trulens.core.feedback import provider as core_provider
from trulens.core.guardrails import base as core_guardrails
from trulens.core.schema import app as app_schema
from trulens.core.schema import base as base_schema
from trulens.core.schema import dataset as dataset_schema
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import groundtruth as groundtruth_schema
from trulens.core.schema import record as record_schema
from trulens.core.schema import select as select_schema
from trulens.core.schema import types as types_schema
from trulens.core.utils import asynchro as asynchro_utils
from trulens.core.utils import constants as constant_utils
from trulens.core.utils import containers as container_utils
from trulens.core.utils import deprecation as deprecation_utils
from trulens.core.utils import imports as import_utils
from trulens.core.utils import json as json_utils
from trulens.core.utils import keys as keys_utils
from trulens.core.utils import pace as pace_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import text as text_utils
from trulens.core.utils import threading as threading_utils
from trulens.core.utils import trulens as trulens_utils
from trulens.dashboard import (
    Leaderboard as Leaderboard_page,  # not a good location
)
from trulens.dashboard import appui as dashboard_appui
from trulens.dashboard import constants as dashboard_constants
from trulens.dashboard import display as dashboard_display
from trulens.dashboard import run as dashboard_run
from trulens.dashboard import streamlit as dashboard_streamlit
from trulens.dashboard.pages import Compare as Compare_page
from trulens.dashboard.pages import Records as Records_page
from trulens.dashboard.utils import dashboard_utils
from trulens.dashboard.utils import metadata_utils
from trulens.dashboard.utils import notebook_utils
from trulens.dashboard.utils import records_utils
from trulens.dashboard.ux import components as dashboard_components
from trulens.dashboard.ux import styles as dashboard_styles
from trulens.experimental.otel_tracing.core._utils import wrap as wrap_utils
from trulens.feedback import embeddings as feedback_embeddings
from trulens.feedback import feedback as mod_feedback
from trulens.feedback import generated as feedback_generated
from trulens.feedback import groundtruth as feedback_groundtruth
from trulens.feedback import llm_provider
from trulens.feedback import prompts as feedback_prompts
from trulens.feedback.dummy import endpoint as dummy_endpoint
from trulens.feedback.dummy import provider as dummy_provider
from trulens.feedback.v2 import feedback as mod_feedback_v2  # not a good name
from trulens.feedback.v2.provider import (
    base as core_v2_provider,  # not a good name
)
from trulens.providers.bedrock import endpoint as bedrock_endpoint
from trulens.providers.bedrock import provider as bedrock_provider
from trulens.providers.cortext import endpoint as cortex_endpoint
from trulens.providers.cortext import provider as cortex_provider
from trulens.providers.huggingface import endpoint as huggingface_endpoint
from trulens.providers.huggingface import provider as huggingface_provider
from trulens.providers.langchain import endpoint as langchain_endpoint
from trulens.providers.langchain import provider as langchain_provider
from trulens.providers.litellm import endpoint as litellm_endpoint
from trulens.providers.litellm import provider as litellm_provider
from trulens.providers.openai import endpoint as openai_endpoint
from trulens.providers.openai import provider as openai_provider

# modules without renames that can be imported:
"""
./benchmark/trulens/benchmark/benchmark_frameworks/dataset/beir_loader.py
./benchmark/trulens/benchmark/benchmark_frameworks/experiments/dataset_preprocessing.py
./benchmark/trulens/benchmark/benchmark_frameworks/tru_benchmark_experiment.py
./benchmark/trulens/benchmark/generate/generate_test_set.py
./benchmark/trulens/benchmark/test_cases.py
./connectors/snowflake/trulens/connectors/snowflake/utils/server_side_evaluation_stored_procedure.py
./connectors/snowflake/trulens/connectors/snowflake/utils/server_side_evaluation_artifacts.py
"""

# modules that should not be imported at all
"""
./trulens/_mods.py
./core/trulens/core/database/migrations/env.py
./core/trulens/core/database/migrations/versions/2_add_run_location_column_to_feedback_defs_table.py
./core/trulens/core/database/migrations/versions/9_update_app_json.py
./core/trulens/core/database/migrations/versions/5_add_app_name_and_version_fields.py
./core/trulens/core/database/migrations/versions/3_add_groundtruth_and_dataset_tables.py
./core/trulens/core/database/migrations/versions/8_update_records_app_id.py
./core/trulens/core/database/migrations/versions/4_set_ff_id_not_null.py
./core/trulens/core/database/migrations/versions/7_app_name_version_not_null.py
./core/trulens/core/database/migrations/versions/6_populate_app_name_and_version_data.py
./core/trulens/core/database/migrations/versions/1_first_revision.py
"""
