import logging
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple
import warnings

import pydantic

from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.base import OutputType
from trulens_eval.feedback.provider.base import WithOutputType
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.feedback.provider.endpoint.base import EndpointDelayError
from trulens_eval.utils.generated import re_0_10_rating
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LAMINI

with OptionalImports(messages=REQUIREMENT_LAMINI):
    import lamini

    from trulens_eval.feedback.provider.endpoint.lamini import LaminiEndpoint

# check that the optional imports are not dummies:
OptionalImports(messages=REQUIREMENT_LAMINI).assert_installed(lamini)

logger = logging.getLogger(__name__)


class Lamini(WithOutputType, LLMProvider):
    """Out of the box feedback functions calling Lamini API.

    Create an Lamini Provider with out of the box feedback functions. Lamini
    supports output type specification making it more efficient at some
    tasks/feedback functions.

    List of supported models can be found on the [Lamini model list
    page](https://lamini-ai.github.io/inference/models_list/).

    Usage:
        ```python
        from trulens_eval.feedback.provider.lamini import Lamini
        lamini_provider = Lamini()
        ```
    """

    DEFAULT_MODEL_NAME: ClassVar[str] = "mistralai/Mistral-7B-Instruct-v0.1"
    """Default model name."""

    model_engine: str = pydantic.Field(alias="model_name")
    """Model specification of parent class.
    
    We alias to `model_name` to match lamini terminology.
    """

    model_name: str = DEFAULT_MODEL_NAME
    """The Lamini completion model. 
    
    Defaults to `mistralai/Mistral-7B-Instruct-v0.1`.
    
    List can be found on the [Lamini model list
    page](https://lamini-ai.github.io/inference/models_list/).
    """

    generation_args: Dict[str, Any] = pydantic.Field(default_factory=dict)
    """Additional arguments to pass to the `Lamini.generate` as needed for
    model/usage.

    Warning:
        Feedback functions override the `output_type` argument to
        `Lamini.generate` so this parameter cannot be set using
        `generation_args`.
    """

    endpoint: Endpoint

    def __init__(
        self,
        model_name: Optional[str] = None,
        generation_kwargs: Optional[Dict] = None,
        endpoint: Optional[Endpoint] = None,
        **kwargs: dict
    ):
        # NOTE(piotrm): HACK006: pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.

        if model_name is None:
            model_name = self.DEFAULT_MODEL_NAME

        if generation_kwargs is None:
            generation_kwargs = {}

        if 'output_type' in generation_kwargs:
            raise ValueError(
                "`output_type` cannot be set for `generation_args` as it is overwritten by each feedback function."
            )

        self_kwargs = {}
        self_kwargs.update(**kwargs)
        self_kwargs['model_name'] = model_name
        self_kwargs['generation_args'] = generation_kwargs
        self_kwargs['endpoint'] = LaminiEndpoint(
            **kwargs
        )

        if lamini.api_key is None: #  and os.environ.get("LAMINI_API_KEY") is None:
            logger.warning(
                "No lamini API key is set. "
                "You may need to set lamini.api_key before using this provider."
            )

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__

    def generate_score_and_reasons(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        normalize: float = 10.0
    ) -> Tuple[float, Dict]:
        """
        Method to generate a score and reasons. This method operates only on
        single expected fill of the template so splitting needs to be done
        outside of this method.

        Args:
            system_prompt: A pre-formated system prompt.

            normalize: A float to normalize the score with. If the prompt asks a
                generation in range [0, X], normalize should be X.

        Returns:
            The score on 0-1 scale and reason metadata (dict) if returned by the
                LLM.
        """
        assert self.endpoint is not None, "Endpoint is not set."

        llm_messages = [{"role": "system", "content": system_prompt}]
        if user_prompt is not None:
            llm_messages.append({"role": "user", "content": user_prompt})

        response = self.endpoint.run_in_pace(
            func=self.create_chat_completion, messages=llm_messages
        )
        if "Supporting Evidence" in response:
            score = -1
            supporting_evidence = None
            criteria = None
            for line in response.split('\n'):
                if "Score" in line:
                    score = re_0_10_rating(line) / normalize
                criteria_lines = []
                supporting_evidence_lines = []
                collecting_criteria = False
                collecting_evidence = False

                for line in response.split('\n'):
                    if "Criteria:" in line:
                        criteria_lines.append(
                            line.split("Criteria:", 1)[1].strip()
                        )
                        collecting_criteria = True
                        collecting_evidence = False
                    elif "Supporting Evidence:" in line:
                        supporting_evidence_lines.append(
                            line.split("Supporting Evidence:", 1)[1].strip()
                        )
                        collecting_evidence = True
                        collecting_criteria = False
                    elif collecting_criteria:
                        if "Supporting Evidence:" not in line:
                            criteria_lines.append(line.strip())
                        else:
                            collecting_criteria = False
                    elif collecting_evidence:
                        if "Criteria:" not in line:
                            supporting_evidence_lines.append(line.strip())
                        else:
                            collecting_evidence = False

                criteria = "\n".join(criteria_lines).strip()
                supporting_evidence = "\n".join(supporting_evidence_lines
                                               ).strip()
            reasons = {
                'reason':
                    (
                        f"{'Criteria: ' + str(criteria)}\n"
                        f"{'Supporting Evidence: ' + str(supporting_evidence)}"
                    )
            }
            return score, reasons

        else:
            score = re_0_10_rating(response) / normalize
            warnings.warn(
                "No supporting evidence provided. Returning score only.",
                UserWarning
            )
            return score, {}
        

    def create_chat_completion_with_output_type(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        output_type: Optional[Dict[str, OutputType]] = None,
        **kwargs
    ) -> str:

        if lamini.api_key is None:
            raise ValueError(
                "Lamini API key is not set. "
                "Please set `lamini.api_key` before using the lamini provider."
            )

        lamini_instance = lamini.Lamini(model_name=self.model_name)

        if output_type is None:
            output_type = {'output': "string"}

        if prompt is not None:
            pass
        elif messages is not None:
            # Assume there is only one system message.
            if len(messages) > 1:
                raise ValueError(
                    "Lamini only supports a single system message in a single completion."
                )
            prompt=messages[0]['content']
        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        all_args = dict(
            output_type=output_type,
            **kwargs,
            **self.generation_args
        )

        def invoke_endpoint(prompt, **args):
            try:
                comp = lamini_instance.generate(prompt=prompt, **args)

            except lamini.error.APIError as e:
                if "Please try again in a few minutes." in str(e):
                    raise EndpointDelayError(delay=60) from e
                raise e

            return comp

        comp = self.endpoint.run_in_pace(invoke_endpoint, prompt=prompt, **all_args)

        if any(output_key not in comp for output_key in output_type.keys()):
            raise RuntimeError(
                f"Unexpected response from lamini is missing some/all keys: {comp}. "
                f"Expected keys: {list(output_type.keys())}"
            )

        return comp
