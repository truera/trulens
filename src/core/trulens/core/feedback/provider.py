import logging
from typing import ClassVar, Optional

import trulens.core.feedback.endpoint as mod_endpoint
from trulens.utils.pyschema import WithClassInfo
from trulens.utils.serial import SerialModel

logger = logging.getLogger(__name__)


class Provider(WithClassInfo, SerialModel):
    """Base Provider class.

    TruLens makes use of *Feedback Providers* to generate evaluations of
    large language model applications. These providers act as an access point
    to different models, most commonly classification models and large language models.

    These models are then used to generate feedback on application outputs or intermediate
    results.

    `Provider` is the base class for all feedback providers. It is an abstract
    class and should not be instantiated directly. Rather, it should be subclassed
    and the subclass should implement the methods defined in this class.

    There are many feedback providers available in TruLens that grant access to a wide range
    of proprietary and open-source models.

    Providers for classification and other non-LLM models should directly subclass `Provider`.
    The feedback functions available for these providers are tied to specific providers, as they
    rely on provider-specific endpoints to models that are tuned to a particular task.

    For example, the Huggingface feedback provider provides access to a number of classification models
    for specific tasks, such as language detection. These models are than utilized by a feedback function
    to generate an evaluation score.

    !!! example

        ```python
        from trulens.ext.provider.huggingface import Huggingface
        huggingface_provider = Huggingface()
        huggingface_provider.language_match(prompt, response)
        ```

    Providers for LLM models should subclass `trulens.external.providers.LLMProvider`, which itself subclasses `Provider`.
    Providers for LLM-generated feedback are more of a plug-and-play variety. This means that the
    base model of your choice can be combined with feedback-specific prompting to generate feedback.

    For example, `relevance` can be run with any base LLM feedback provider. Once the feedback provider
    is instantiated with a base model, the `relevance` function can be called with a prompt and response.

    This means that the base model selected is combined with specific prompting for `relevance` to generate feedback.

    !!! example

        ```python
        from trulens.ext.provider.openai import OpenAI
        provider = OpenAI(model_engine="gpt-3.5-turbo")
        provider.relevance(prompt, response)
        ```
    """

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    endpoint: Optional[mod_endpoint.Endpoint] = None
    """Endpoint supporting this provider.

    Remote API invocations are handled by the endpoint.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
