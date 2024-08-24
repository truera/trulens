# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.core.utils.imports` instead.
"""

import warnings

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.utils.imports import MESSAGE_DEBUG_OPTIONAL_PACKAGE_NOT_FOUND
from trulens.core.utils.imports import MESSAGE_ERROR_REQUIRED_PACKAGE_NOT_FOUND
from trulens.core.utils.imports import MESSAGE_FRAGMENT_VERSION_MISMATCH
from trulens.core.utils.imports import (
    MESSAGE_FRAGMENT_VERSION_MISMATCH_OPTIONAL,
)
from trulens.core.utils.imports import MESSAGE_FRAGMENT_VERSION_MISMATCH_PIP
from trulens.core.utils.imports import (
    MESSAGE_FRAGMENT_VERSION_MISMATCH_REQUIRED,
)
from trulens.core.utils.imports import REQUIREMENT_BERT_SCORE
from trulens.core.utils.imports import REQUIREMENT_EVALUATE
from trulens.core.utils.imports import REQUIREMENT_GROUNDEDNESS
from trulens.core.utils.imports import (
    REQUIREMENT_INSTRUMENT_LLAMA as REQUIREMENT_LLAMA,
)
from trulens.core.utils.imports import (
    REQUIREMENT_INSTRUMENT_NEMO as REQUIREMENT_RAILS,
)
from trulens.core.utils.imports import REQUIREMENT_NOTEBOOK
from trulens.core.utils.imports import REQUIREMENT_PINECONE
from trulens.core.utils.imports import (
    REQUIREMENT_PROVIDER_BEDROCK as REQUIREMENT_BEDROCK,
)
from trulens.core.utils.imports import (
    REQUIREMENT_PROVIDER_CORTEX as REQUIREMENT_CORTEX,
)
from trulens.core.utils.imports import (
    REQUIREMENT_PROVIDER_HUGGINGFACE as REQUIREMENT_HUGGINGFACE,
)
from trulens.core.utils.imports import (
    REQUIREMENT_PROVIDER_LANGCHAIN as REQUIREMENT_LANGCHAIN,
)
from trulens.core.utils.imports import (
    REQUIREMENT_PROVIDER_LITELLM as REQUIREMENT_LITELLM,
)
from trulens.core.utils.imports import (
    REQUIREMENT_PROVIDER_OPENAI as REQUIREMENT_OPENAI,
)
from trulens.core.utils.imports import REQUIREMENT_SKLEARN
from trulens.core.utils.imports import Dummy
from trulens.core.utils.imports import ImportErrorMessages
from trulens.core.utils.imports import OptionalImports
from trulens.core.utils.imports import VersionConflict
from trulens.core.utils.imports import all_packages
from trulens.core.utils.imports import check_imports
from trulens.core.utils.imports import format_import_errors
from trulens.core.utils.imports import get_package_version
from trulens.core.utils.imports import optional_packages
from trulens.core.utils.imports import parse_version
from trulens.core.utils.imports import pin_spec
from trulens.core.utils.imports import required_packages
from trulens.core.utils.imports import requirements_of_file
from trulens.core.utils.imports import static_resource

trulens_name: str = "trulens"
