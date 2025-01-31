"""
Soon to be deprecated in favor of [TruApp][trulens.apps.app.TruApp].
"""

import logging
from pprint import PrettyPrinter
import warnings

from trulens.apps.app import TruApp
from trulens.apps.app import instrument as new_instrument

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

# Keys used in app_extra_json to indicate an automatically added structure for
# places an instrumented method exists but no instrumented data exists
# otherwise.
PLACEHOLDER = "__tru_placeholder"


class TruCustomApp(TruApp):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "TruCustomApp is being deprecated in the next major version; use TruApp instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


warnings.warn(
    """from trulens.apps.custom import instrument
        is being deprecated in the next major version; use from trulens.apps.app import instrument
        instead.""",
    DeprecationWarning,
    stacklevel=2,
)

# Alias instrument to the new location while keeping the old import path working
instrument = new_instrument


TruCustomApp.model_rebuild()
