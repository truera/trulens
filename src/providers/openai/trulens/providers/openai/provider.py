import logging
from typing import (
    Any,
    ClassVar,
    Dict,
    Optional,
    Sequence,
    Type,
    Union,
)

import pydantic
from trulens.core.utils import constants as constant_utils
from trulens.core.utils import pace as pace_utils
from trulens.feedback import llm_provider
from trulens.feedback import output_schemas as feedback_output_schemas
from trulens.providers.openai import endpoint as openai_endpoint

import openai as oai

logger = logging.getLogger(__name__)

SEED: int = 123


# --- Optional CFG grammars matching feedback output schemas ---
# Fixed-field-order JSON grammars for robust constrained generation.

# BaseFeedbackResponse JSON: {"score": <int>}
BASE_FEEDBACK_RESPONSE_LARK_GRAMMAR: str = r"""
%import common.INT
%import common.WS
%ignore WS

start: "{" "\"score\"" ":" INT "}"
"""

# ChainOfThoughtResponse JSON:
# {"criteria":"...","supporting_evidence":"...","score":<int>}
CHAIN_OF_THOUGHT_RESPONSE_LARK_GRAMMAR: str = r"""
%import common.ESCAPED_STRING
%import common.INT
%import common.WS
%ignore WS

start: "{" "\"criteria\"" ":" ESCAPED_STRING "," "\"supporting_evidence\"" ":" ESCAPED_STRING "," "\"score\"" ":" INT "}"
"""


class OpenAI(llm_provider.LLMProvider):
    """Out of the box feedback functions calling OpenAI APIs.

    Additionally, all feedback functions listed in the base [LLMProvider
    class][trulens.feedback.LLMProvider] can be run with OpenAI.

    Create an OpenAI Provider with out of the box feedback functions.

    Example:
        ```python
        from trulens.providers.openai import OpenAI
        openai_provider = OpenAI()
        ```

    Args:
        model_engine: The OpenAI completion model. Defaults to
            `gpt-4o-mini`

        **kwargs: Additional arguments to pass to the
            [OpenAIEndpoint][trulens.providers.openai.endpoint.OpenAIEndpoint]
            which are then passed to
            [OpenAIClient][trulens.providers.openai.endpoint.OpenAIClient]
            and finally to the OpenAI client.
    """

    DEFAULT_MODEL_ENGINE: ClassVar[str] = "gpt-4o-mini"

    # Endpoint cannot presently be serialized but is constructed in __init__
    # below so it is ok.
    endpoint: openai_endpoint.OpenAIEndpoint = pydantic.Field(exclude=True)

    def __init__(
        self,
        *args,
        endpoint=None,
        pace: Optional[pace_utils.Pace] = None,
        rpm: Optional[int] = None,
        model_engine: Optional[str] = None,
        **kwargs: dict,
    ):
        # NOTE(piotrm): HACK006: pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.

        if model_engine is None:
            model_engine = self.DEFAULT_MODEL_ENGINE

        # Separate set of args for our attributes because only a subset go into
        # endpoint below.
        self_kwargs: Dict[str, Any] = dict()
        self_kwargs.update(**kwargs)
        self_kwargs["model_engine"] = model_engine

        self_kwargs["endpoint"] = openai_endpoint.OpenAIEndpoint(
            *args, pace=pace, rpm=rpm, **kwargs
        )

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__

    def _structured_output_supported(self) -> bool:
        """Whether the provider supports structured output. This is analogous to model support for OpenAI's Responses API.
        For more details: https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#structured-outputs-vs-json-mode
        """
        # Reasoning models have limited structured output support
        if self._is_reasoning_model():
            return False

        if (
            # gpt-3.5, gpt-3.5-turbo do not support structured output
            self.model_engine.startswith("gpt-3.5")
            # gpt-4, gpt-4-turbo do not support structured output
            or (
                self.model_engine.startswith("gpt-4")
                and not self.model_engine.startswith("gpt-4o")
            )
            # gpt-4o-2024-05-13 does not support structured output
            or self.model_engine == "gpt-4o-2024-05-13"
            # NOTE (corey, 2025-06-30): Unclear if deep-research will support structured output in the future.
            or self.model_engine.endswith("-deep-research")
        ):
            return False
        return True

    # --- Capability probing and caching helpers ---
    def _capabilities_key(self) -> str:
        return self.model_engine

    def _is_cfg_available(self) -> bool:
        """Return True if model supports CFG path by default heuristics.

        Currently enables CFG only for gpt-5* model families.
        """
        return self.model_engine.startswith("gpt-5")

    def _call_with_capability_fallbacks(
        self,
        *,
        input_messages: Sequence[Dict[str, Any]],
        response_format: Optional[Type[pydantic.BaseModel]],
        kwargs: Dict[str, Any],
    ) -> Optional[Union[str, pydantic.BaseModel]]:
        capabilities = self._get_capabilities()

        # 0) Grammar-constrained generation (CFG) via Responses.create if provided
        grammar_syntax: Optional[str] = kwargs.pop("grammar_syntax", None)
        grammar_definition: Optional[str] = kwargs.pop(
            "grammar_definition", None
        )
        grammar_name: str = kwargs.pop("grammar_name", "custom_grammar")
        grammar_description: str = kwargs.pop(
            "grammar_description", "Custom grammar-constrained generation."
        )

        cfg_capability = capabilities.get("cfg")
        if (
            grammar_syntax
            and grammar_definition
            and (cfg_capability is None or cfg_capability is True)
        ):
            try:
                # Filter params not supported by responses.create (e.g., reasoning_effort)
                _responses_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ("model", "reasoning_effort", "seed")
                }
                response = self.endpoint.client.responses.create(
                    model=kwargs.get("model", self.model_engine),
                    input=input_messages,
                    text={"format": {"type": "text"}},
                    tools=[
                        {
                            "type": "custom",
                            "name": grammar_name,
                            "description": grammar_description,
                            "format": {
                                "type": "grammar",
                                "syntax": grammar_syntax,
                                "definition": grammar_definition,
                            },
                        }
                    ],
                    parallel_tool_calls=False,
                    **_responses_kwargs,
                )

                # Mark CFG as supported for this model
                self._set_capabilities({"cfg": True})

                if hasattr(response, "output"):
                    out_items = getattr(response, "output")
                    for item in out_items:
                        if getattr(item, "type", None) == "tool" and hasattr(
                            item, "input"
                        ):
                            return getattr(item, "input")
                    texts: list[str] = []
                    for item in out_items:
                        for part in getattr(item, "content", []) or []:
                            if getattr(
                                part, "type", None
                            ) == "output_text" and hasattr(part, "text"):
                                texts.append(part.text)
                    if texts:
                        return "".join(texts)

                try:
                    return response.model_dump_json()
                except Exception:
                    return str(response)
            except Exception as exc:
                # Mark CFG as unsupported to avoid re-trying on subsequent calls
                self._set_capabilities({"cfg": False})
                logger.debug(
                    f"[TruLens] CFG grammar invocation failed for model '{self.model_engine}': {exc}. Falling back."
                )

        # 1) Try structured outputs via Responses API if requested/unknown
        wants_structured_outputs = response_format is not None
        structured_outputs = (
            capabilities.get("structured_outputs")
            if "structured_outputs" in capabilities
            else None
        )
        if wants_structured_outputs and (
            structured_outputs is None or structured_outputs is True
        ):
            try:
                response = self.endpoint.client.responses.parse(
                    input=input_messages, text_format=response_format, **kwargs
                )
                self._set_capabilities({"structured_outputs": True})
                return response.output_parsed
            except Exception as exc:
                # Targeted retry: remove only offending params and retry once
                offending_params = []
                for p in ("seed", "temperature", "reasoning_effort"):
                    try:
                        if (
                            self._is_unsupported_parameter_error(exc, p)
                            and p in kwargs
                        ):
                            offending_params.append(p)
                    except Exception:
                        pass

                if offending_params:
                    for p in offending_params:
                        kwargs.pop(p, None)
                    try:
                        response = self.endpoint.client.responses.parse(
                            input=input_messages,
                            text_format=response_format,
                            **kwargs,
                        )
                        self._set_capabilities({"structured_outputs": True})
                        return response.output_parsed
                    except Exception as exc2:
                        if (
                            self._is_unsupported_parameter_error(
                                exc2, "response_format"
                            )
                            or "structured" in str(exc2).lower()
                            or "responses.parse" in str(exc2).lower()
                        ):
                            self._set_capabilities({
                                "structured_outputs": False
                            })
                            logger.debug(
                                f"[TruLens] Structured outputs unsupported for model '{self.model_engine}'. Falling back to text outputs."
                            )
                        else:
                            raise
                else:
                    if (
                        self._is_unsupported_parameter_error(
                            exc, "response_format"
                        )
                        or "structured" in str(exc).lower()
                        or "responses.parse" in str(exc).lower()
                    ):
                        self._set_capabilities({"structured_outputs": False})
                        logger.debug(
                            f"[TruLens] Structured outputs unsupported for model '{self.model_engine}'. Falling back to text outputs."
                        )
                    else:
                        raise

        # 2) Fall back to Chat Completions with parameter probes
        # Probe temperature support
        if "temperature" in kwargs:
            temperature = capabilities.get("temperature")
            if temperature is False:
                kwargs.pop("temperature", None)
            elif temperature is None:
                try:
                    completion = self.endpoint.client.chat.completions.create(
                        messages=input_messages, **kwargs
                    )
                    self._set_capabilities({"temperature": True})
                    return completion.choices[0].message.content
                except Exception as exc:
                    if self._is_unsupported_parameter_error(exc, "temperature"):
                        kwargs.pop("temperature", None)
                        self._set_capabilities({"temperature": False})
                        # Immediate retry without temperature
                        completion = (
                            self.endpoint.client.chat.completions.create(
                                messages=input_messages, **kwargs
                            )
                        )
                        return completion.choices[0].message.content
                    else:
                        raise

        # Probe reasoning_effort support
        if "reasoning_effort" in kwargs:
            reasoning_effort = capabilities.get("reasoning_effort")
            if reasoning_effort is False:
                kwargs.pop("reasoning_effort", None)
            elif reasoning_effort is None:
                try:
                    completion = self.endpoint.client.chat.completions.create(
                        messages=input_messages, **kwargs
                    )
                    self._set_capabilities({"reasoning_effort": True})
                    return completion.choices[0].message.content
                except Exception as exc:
                    if self._is_unsupported_parameter_error(
                        exc, "reasoning_effort"
                    ):
                        kwargs.pop("reasoning_effort", None)
                        self._set_capabilities({"reasoning_effort": False})
                        # Immediate retry without reasoning_effort
                        completion = (
                            self.endpoint.client.chat.completions.create(
                                messages=input_messages, **kwargs
                            )
                        )
                        return completion.choices[0].message.content
                    else:
                        raise

        # Final attempt with whatever parameters remain
        completion = self.endpoint.client.chat.completions.create(
            messages=input_messages, **kwargs
        )
        return completion.choices[0].message.content

    # LLMProvider requirement
    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        response_format: Optional[Type[pydantic.BaseModel]] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ) -> Optional[Union[str, pydantic.BaseModel]]:
        if "model" not in kwargs:
            kwargs["model"] = self.model_engine

        if messages is not None:
            input_messages = messages
        elif prompt is not None:
            input_messages = [{"role": "system", "content": prompt}]
        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        # Handle reasoning models
        if self._is_reasoning_model():
            # Reasoning models don't support temperature parameter
            if "temperature" in kwargs:
                logger.warning(
                    f"Temperature parameter is not supported for reasoning model {self.model_engine}. "
                    "Removing temperature parameter."
                )
                del kwargs["temperature"]

            # Add reasoning_effort parameter if provided
            if reasoning_effort is not None:
                if reasoning_effort not in ["low", "medium", "high"]:
                    logger.warning(
                        f"Invalid reasoning_effort '{reasoning_effort}'. Must be 'low', 'medium', or 'high'. Using 'medium' as default."
                    )
                else:
                    # For reasoning models, use the reasoning_effort parameter directly
                    kwargs["reasoning_effort"] = reasoning_effort
        else:
            # Set default temperature for non-reasoning models
            if "temperature" not in kwargs:
                kwargs["temperature"] = 0.0

        # Route through capability probing + caching for robust behavior
        if "seed" not in kwargs:
            kwargs["seed"] = SEED

        # Auto‑enable CFG for feedback schemas when available (gpt-5*)
        try:
            if self._is_cfg_available() and isinstance(response_format, type):
                if (
                    response_format
                    is feedback_output_schemas.BaseFeedbackResponse
                ):
                    kwargs["grammar_syntax"] = "lark"
                    kwargs["grammar_definition"] = (
                        BASE_FEEDBACK_RESPONSE_LARK_GRAMMAR
                    )
                    kwargs["grammar_name"] = "base_feedback_json"
                    kwargs["grammar_description"] = (
                        "Strict JSON for BaseFeedbackResponse"
                    )
                elif (
                    response_format
                    is feedback_output_schemas.ChainOfThoughtResponse
                ):
                    kwargs["grammar_syntax"] = "lark"
                    kwargs["grammar_definition"] = (
                        CHAIN_OF_THOUGHT_RESPONSE_LARK_GRAMMAR
                    )
                    kwargs["grammar_name"] = "cot_feedback_json"
                    kwargs["grammar_description"] = (
                        "Strict JSON for ChainOfThoughtResponse"
                    )
        except Exception as exc:
            logger.debug(
                f"[TruLens] Auto-enable CFG grammar setup failed for model '{self.model_engine}' and response_format '{type(response_format).__name__}': {exc}",
                exc_info=True,
            )

        return self._call_with_capability_fallbacks(
            input_messages=input_messages,
            response_format=response_format,
            kwargs=kwargs,
        )

    def _moderation(self, text: str):
        # See https://platform.openai.com/docs/guides/moderation/overview .
        moderation_response = self.endpoint.run_in_pace(
            func=self.endpoint.client.moderations.create, input=text
        )
        return moderation_response.results[0]

    # TODEP
    def moderation_hate(self, text: str) -> float:
        """A function that checks if text is hate speech.

        Example:
            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI
            openai_provider = OpenAI()

            feedback = Feedback(
                openai_provider.moderation_hate, higher_is_better=False
            ).on_output()
            ```

        Args:
            text: Text to evaluate.

        Returns:
            float: A value between 0.0 (not hate) and 1.0 (hate).
        """
        openai_response = self._moderation(text)
        return float(openai_response.category_scores.hate)

    # TODEP
    def moderation_hatethreatening(self, text: str) -> float:
        """A function that checks if text is threatening speech.

        Example:
            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI
            openai_provider = OpenAI()

            feedback = Feedback(
                openai_provider.moderation_hatethreatening, higher_is_better=False
            ).on_output()
            ```

        Args:
            text: Text to evaluate.

        Returns:
            float: A value between 0.0 (not threatening) and 1.0 (threatening).
        """
        openai_response = self._moderation(text)

        return float(openai_response.category_scores.hate_threatening)

    # TODEP
    def moderation_selfharm(self, text: str) -> float:
        """A function that checks if text is about self harm.

        Example:
            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI
            openai_provider = OpenAI()

            feedback = Feedback(
                openai_provider.moderation_selfharm, higher_is_better=False
            ).on_output()
            ```

        Args:
            text: Text to evaluate.

        Returns:
            float: A value between 0.0 (not self harm) and 1.0 (self harm).
        """
        openai_response = self._moderation(text)

        return float(openai_response.category_scores.self_harm)

    # TODEP
    def moderation_sexual(self, text: str) -> float:
        """A function that checks if text is sexual speech.

        Example:
            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI
            openai_provider = OpenAI()

            feedback = Feedback(
                openai_provider.moderation_sexual, higher_is_better=False
            ).on_output()
            ```

        Args:
            text: Text to evaluate.

        Returns:
            float: A value between 0.0 (not sexual) and 1.0 (sexual).
        """
        openai_response = self._moderation(text)

        return float(openai_response.category_scores.sexual)

    # TODEP
    def moderation_sexualminors(self, text: str) -> float:
        """A function that checks if text is about sexual minors.

        Example:
            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI
            openai_provider = OpenAI()

            feedback = Feedback(
                openai_provider.moderation_sexualminors, higher_is_better=False
            ).on_output()
            ```

        Args:
            text: Text to evaluate.

        Returns:
            float: A value between 0.0 (not sexual minors) and 1.0 (sexual minors).
        """

        openai_response = self._moderation(text)

        return float(openai_response.category_scores.sexual_minors)

    # TODEP
    def moderation_violence(self, text: str) -> float:
        """A function that checks if text is about violence.

        Example:
            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI
            openai_provider = OpenAI()

            feedback = Feedback(
                openai_provider.moderation_violence, higher_is_better=False
            ).on_output()
            ```

        Args:
            text: Text to evaluate.

        Returns:
            float: A value between 0.0 (not violence) and 1.0 (violence).
        """
        openai_response = self._moderation(text)

        return float(openai_response.category_scores.violence)

    # TODEP
    def moderation_violencegraphic(self, text: str) -> float:
        """A function that checks if text is about graphic violence.

        Example:
            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI
            openai_provider = OpenAI()

            feedback = Feedback(
                openai_provider.moderation_violencegraphic, higher_is_better=False
            ).on_output()
            ```

        Args:
            text: Text to evaluate.

        Returns:
            float: A value between 0.0 (not graphic violence) and 1.0 (graphic violence).
        """
        openai_response = self._moderation(text)

        return float(openai_response.category_scores.violence_graphic)

    # TODEP
    def moderation_harassment(self, text: str) -> float:
        """A function that checks if text is about graphic violence.

        Example:
            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI
            openai_provider = OpenAI()

            feedback = Feedback(
                openai_provider.moderation_harassment, higher_is_better=False
            ).on_output()
            ```

        Args:
            text: Text to evaluate.

        Returns:
            float: A value between 0.0 (not harassment) and 1.0 (harassment).
        """
        openai_response = self._moderation(text)

        return float(openai_response.category_scores.harassment)

    def moderation_harassment_threatening(self, text: str) -> float:
        """A function that checks if text is about graphic violence.

        Example:
            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI
            openai_provider = OpenAI()

            feedback = Feedback(
                openai_provider.moderation_harassment_threatening, higher_is_better=False
            ).on_output()
            ```

        Args:
            text: Text to evaluate.

        Returns:
            float: A value between 0.0 (not harassment/threatening) and 1.0 (harassment/threatening).
        """
        openai_response = self._moderation(text)

        return float(openai_response.category_scores.harassment)


class AzureOpenAI(OpenAI):
    """
    !!! warning
        _Azure OpenAI_ does not support the _OpenAI_ moderation endpoint.
    Out of the box feedback functions calling AzureOpenAI APIs. Has the same
    functionality as OpenAI out of the box feedback functions, excluding the
    moderation endpoint which is not supported by Azure. Please export the
    following env variables. These can be retrieved from https://oai.azure.com/
    .

    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_API_KEY
    - OPENAI_API_VERSION

    Deployment name below is also found on the oai azure page.

    Example:
        ```python
        from trulens.providers.openai import AzureOpenAI
        openai_provider = AzureOpenAI(deployment_name="...")

        openai_provider.relevance(
            prompt="Where is Germany?",
            response="Poland is in Europe."
        ) # low relevance
        ```

    Args:
        deployment_name: The name of the deployment.
    """

    # Sent to our openai client wrapper but need to keep here as well so that it
    # gets dumped when jsonifying.
    deployment_name: str = pydantic.Field(alias="model_engine")

    def __init__(
        self,
        deployment_name: str,
        endpoint: Optional[openai_endpoint.OpenAIEndpoint] = None,
        **kwargs: dict,
    ):
        # NOTE(piotrm): HACK006: pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.

        # Make a dict of args to pass to AzureOpenAI client. Remove any we use
        # for our needs. Note that model name / deployment name is not set in
        # that client and instead is an argument to each chat request. We pass
        # that through the super class's `_create_chat_completion`.
        client_kwargs = dict(kwargs)
        if constant_utils.CLASS_INFO in client_kwargs:
            del client_kwargs[constant_utils.CLASS_INFO]

        if "model_engine" in client_kwargs:
            # delete from client args
            del client_kwargs["model_engine"]
        else:
            # but include in provider args
            kwargs["model_engine"] = deployment_name

        kwargs["client"] = openai_endpoint.OpenAIClient(
            client=oai.AzureOpenAI(**client_kwargs)
        )

        super().__init__(
            endpoint=None, **kwargs
        )  # need to include pydantic.BaseModel.__init__

    def _create_chat_completion(self, *args, **kwargs):
        """
        We need to pass `engine`
        """
        return super()._create_chat_completion(*args, **kwargs)
