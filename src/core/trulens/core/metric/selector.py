# Re-export from feedback.selector for now.
# This ensures Selector is available from the metric module.
# The actual implementation remains in feedback/selector.py to avoid duplication.

from trulens.core.feedback.selector import ProcessedContentNode
from trulens.core.feedback.selector import Selector
from trulens.core.feedback.selector import Trace

__all__ = [
    "Selector",
    "Trace",
    "ProcessedContentNode",
]
