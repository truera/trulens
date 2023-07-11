import logging

from trulens_eval import app

logger = logging.getLogger(__name__)

for attr in dir(app):
    if not attr.startswith("_"):
        globals()[attr] = getattr(app, attr)

# Since 0.2.0
logger.warning(
    "`trulens_eval.tru_app` is deprecated, use `trulens_eval.app` instead."
)
