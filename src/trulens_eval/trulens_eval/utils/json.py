# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens.core.utils.json`
    or `trulens.core.utils.pyschema` instead.
"""

import warnings

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.utils.json import ALL_SPECIAL_KEYS
from trulens.core.utils.json import CIRCLE
from trulens.core.utils.json import CLASS_INFO
from trulens.core.utils.json import JSON_BASES
from trulens.core.utils.json import encode_httpx_url
from trulens.core.utils.json import encode_openai_timeout
from trulens.core.utils.json import json_default
from trulens.core.utils.json import json_str_of_obj
from trulens.core.utils.json import jsonify
from trulens.core.utils.json import jsonify_for_ui
from trulens.core.utils.json import obj_id_of_obj
from trulens.core.utils.pyschema import ERROR
from trulens.core.utils.pyschema import NOSERIO
