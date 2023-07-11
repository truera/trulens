import logging

from trulens_eval import db

logger = logging.getLogger(__name__)

for attr in dir(db):
    if not attr.startswith("_"):
        globals()[attr] = getattr(db, attr)

# Since 0.2.0
logger.warning(
    "`trulens_eval.tru_db` is deprecated, use `trulens_eval.db` instead."
)
