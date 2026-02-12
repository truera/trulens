"""
BACKWARD-COMPATIBILITY SHIM.

All template classes have moved to ``trulens.feedback.templates``.
This module re-exports them so that existing imports of the form
``from trulens.feedback.v2.feedback import Groundedness`` keep
working.

Prefer importing from ``trulens.feedback.templates`` in new code.
"""

# ruff: noqa: F401, F403
from trulens.feedback.templates.agent import *
from trulens.feedback.templates.base import *

# Re-export the ``Feedback`` alias explicitly so that
# ``from trulens.feedback.v2.feedback import Feedback`` works.
from trulens.feedback.templates.base import Feedback
from trulens.feedback.templates.base import FeedbackTemplate
from trulens.feedback.templates.quality import *
from trulens.feedback.templates.rag import *
from trulens.feedback.templates.safety import *
