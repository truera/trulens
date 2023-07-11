import logging

from trulens_eval import feedback

logger = logging.getLogger(__name__)

for attr in dir(feedback):
    if not attr.startswith("_"):
        globals()[attr] = getattr(feedback, attr)

# Since 0.2.0
logger.warning(
    "`trulens_eval.tru_feedback` is deprecated, use `trulens_eval.feedback` instead."
)
