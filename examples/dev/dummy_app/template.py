import logging

from trulens.apps.custom import instrument

from examples.dev.dummy_app.dummy import Dummy

logger = logging.getLogger(__name__)


class DummyTemplate(Dummy):
    """Dummy template class that fills a question and context into a template
    that has placeholders for these."""

    def __init__(self, template: str, **kwargs):
        super().__init__(**kwargs)

        self.template = template

    @instrument
    def fill(self, question: str, context: str) -> str:
        """Fill in the template with the question and answer.

        Args:
            question: The question to fill in.

            context: The context to fill in.
        """

        return self.template.replace("{question}", question).replace(
            "{context}", context
        )
