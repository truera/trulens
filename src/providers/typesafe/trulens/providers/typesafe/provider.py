"""Non-generative TypeSafe Jev feedback provider."""

from __future__ import annotations

import hashlib
import json
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Tuple, Union

import pydantic
from trulens.core.feedback import provider as core_provider
from trulens.providers.typesafe import endpoint as typesafe_endpoint


def _stable_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for JSON-like data."""

    serialized = json.dumps(
        value,
        default=str,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def _model_dump(value: Any) -> Dict[str, Any]:
    """Convert an SDK model to a plain dictionary."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return dict(value)
    result = {}
    for field in (
        "type",
        "noul",
        "choice",
        "score",
        "legend",
        "probabilities",
        "confidence",
    ):
        if hasattr(value, field):
            result[field] = getattr(value, field)
    return result


class TypeSafeDecision(pydantic.BaseModel):
    """One typed Jev decision with replay and calibration provenance."""

    model_config = pydantic.ConfigDict(extra="forbid")

    value: Union[float, str]
    probabilities: Dict[str, float]
    confidence: Optional[float] = None
    model: str
    primitive: str
    question_template_hash: str
    ordered_criteria: List[Tuple[str, Optional[str]]]
    state_hash: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    raw_answer: Dict[str, Any]


class TypeSafe(core_provider.Provider):
    """Feedback provider backed by TypeSafe's non-generative Jev model.

    Jev evaluates all questions in one request against the same state. Use
    :meth:`evaluate` to retain typed distributions and provenance, or the
    convenience feedback methods when a bare float is required by TruLens.

    Args:
        model_engine: Concrete Jev model version. Defaults to the reproducible
            pinned version ``jev-1.13.0``.
        client: Optional preconfigured ``TypeSafeClient``. Primarily useful
            for custom transports and tests.
        api_key: Optional API key passed to the official SDK. If omitted, the
            SDK reads ``TYPESAFE_API_KEY``.
        timeout: Request timeout passed to the official SDK.
    """

    DEFAULT_MODEL_ENGINE: ClassVar[str] = "jev-1.13.0"

    model_engine: str
    endpoint: typesafe_endpoint.TypeSafeEndpoint = pydantic.Field(exclude=True)

    def __init__(
        self,
        model_engine: str = DEFAULT_MODEL_ENGINE,
        client: Optional[Any] = None,
        endpoint: Optional[typesafe_endpoint.TypeSafeEndpoint] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        if endpoint is None:
            endpoint = typesafe_endpoint.TypeSafeEndpoint(
                model_engine=model_engine,
                client=client,
                api_key=api_key,
                timeout=timeout,
            )
        super().__init__(
            name="typesafe",
            model_engine=model_engine,
            endpoint=endpoint,
            **kwargs,
        )

    @property
    def client(self) -> Any:
        """Return the official SDK client owned by this provider's endpoint."""
        return self.endpoint.client

    @staticmethod
    def _question_metadata(question: Any) -> Tuple[str, Any, Any]:
        raw = _model_dump(question)
        primitive = str(raw.get("type") or type(question).__name__).lower()
        instructions = raw.get("instructions")
        criteria = raw.get("criteria")
        return primitive, instructions, criteria

    @staticmethod
    def _ordered_criteria(criteria: Any) -> List[Tuple[str, Optional[str]]]:
        if isinstance(criteria, dict):
            return [(str(key), value) for key, value in criteria.items()]
        if isinstance(criteria, list):
            return [
                (str(index), str(value)) for index, value in enumerate(criteria)
            ]
        return []

    def evaluate(
        self,
        state: Any,
        questions: Mapping[str, Any],
    ) -> Dict[str, TypeSafeDecision]:
        """Evaluate typed questions together and retain raw provenance.

        Args:
            state: Text or structured state shared by all questions.
            questions: Mapping of IDs to official SDK ``Noul``, ``Choice``,
                or ``Score`` questions.

        Returns:
            A decision per question, preserving distributions and replay
            metadata before any score normalization.

        Raises:
            ValueError: If no questions are supplied or the response is
                missing an answer.
        """
        if not questions:
            raise ValueError("questions must not be empty.")
        response = self.endpoint.system_one(
            state=state,
            questions=dict(questions),
            model=self.model_engine,
        )
        answers = getattr(response, "answers", None)
        if answers is None:
            answers = {
                **getattr(response, "nouls", {}),
                **getattr(response, "choices", {}),
                **getattr(response, "scores", {}),
            }
        usage = getattr(response, "usage", None)
        response_model = str(getattr(response, "model", self.model_engine))
        decisions = {}
        for question_id, question in questions.items():
            if question_id not in answers:
                raise ValueError(
                    f"TypeSafe response omitted question {question_id!r}."
                )
            answer = answers[question_id]
            raw_answer = _model_dump(answer)
            primitive, instructions, criteria = self._question_metadata(
                question
            )
            if "noul" in raw_answer:
                value: Union[float, str] = float(raw_answer["noul"])
                probabilities = {
                    "no": 1.0 - value,
                    "yes": value,
                }
            elif "choice" in raw_answer:
                value = str(raw_answer["choice"])
                probabilities = {
                    str(key): float(probability)
                    for key, probability in raw_answer.get(
                        "probabilities", {}
                    ).items()
                }
            elif "score" in raw_answer:
                value = float(raw_answer["score"])
                probabilities = {
                    str(key): float(probability)
                    for key, probability in raw_answer.get(
                        "probabilities", {}
                    ).items()
                }
            else:
                raise ValueError(
                    f"Unsupported TypeSafe answer for {question_id!r}."
                )
            question_template = {
                "type": primitive,
                "instructions": instructions,
                "criteria": criteria,
            }
            decisions[question_id] = TypeSafeDecision(
                value=value,
                probabilities=probabilities,
                confidence=raw_answer.get("confidence"),
                model=response_model,
                primitive=primitive,
                question_template_hash=_stable_hash(question_template),
                ordered_criteria=self._ordered_criteria(criteria),
                state_hash=_stable_hash(state),
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
                raw_answer=raw_answer,
            )
        return decisions

    def noul(
        self,
        state: Any,
        instructions: str,
        criteria: Optional[Any] = None,
    ) -> float:
        """Return Jev's calibrated probability that a statement is true."""
        from typesafe_sdk import Noul

        question_kwargs = {"instructions": instructions}
        if criteria is not None:
            question_kwargs["criteria"] = criteria
        decision = self.evaluate(
            state=state,
            questions={"result": Noul(**question_kwargs)},
        )["result"]
        return float(decision.value)

    def choice(
        self,
        state: Any,
        instructions: str,
        criteria: Mapping[str, Optional[str]],
    ) -> Dict[str, float]:
        """Return Jev's probability distribution over unordered options."""
        from typesafe_sdk import Choice

        decision = self.evaluate(
            state=state,
            questions={
                "result": Choice(
                    instructions=instructions,
                    criteria=dict(criteria),
                )
            },
        )["result"]
        return decision.probabilities

    def score(
        self,
        state: Any,
        instructions: str,
        criteria: List[str],
    ) -> float:
        """Return an ordinal Jev score normalized to the interval [0, 1]."""
        from typesafe_sdk import Score

        if len(criteria) < 2:
            raise ValueError("score criteria must contain at least two levels.")
        decision = self.evaluate(
            state=state,
            questions={
                "result": Score(
                    instructions=instructions,
                    criteria=criteria,
                )
            },
        )["result"]
        return float(decision.value) / (len(criteria) - 1)

    def context_relevance(self, question: str, context: str) -> float:
        """Return the probability that context helps answer a question."""
        return self.noul(
            state={"question": question, "context": context},
            instructions=(
                "Does `context` contain information relevant to answering "
                "`question`?"
            ),
        )

    def groundedness_measure(
        self,
        source: str,
        statements: Union[str, List[str]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Batch atomic groundedness judgments into one Jev request.

        Args:
            source: Source text that should support the statements.
            statements: One statement or pre-split atomic statements.

        Returns:
            Mean support probability and per-statement decision provenance.
        """
        if isinstance(statements, str):
            statements = [statements]
        if not statements or any(
            not statement.strip() for statement in statements
        ):
            raise ValueError("statements must contain non-empty strings.")
        from typesafe_sdk import Noul

        questions = {
            f"statement_{index}": Noul(
                instructions=(
                    f"Does `source` support this statement: {statement}"
                )
            )
            for index, statement in enumerate(statements)
        }
        decisions = self.evaluate(state={"source": source}, questions=questions)
        values = [float(decision.value) for decision in decisions.values()]
        return sum(values) / len(values), {
            "statements": {
                statement: decisions[f"statement_{index}"].model_dump()
                for index, statement in enumerate(statements)
            }
        }

    def context_relevance_with_cot_reasons(self, *args: Any, **kwargs: Any):
        """Raise because Jev cannot generate textual explanations."""
        raise NotImplementedError(
            "TypeSafe Jev is non-generative; use context_relevance instead."
        )

    def groundedness_measure_with_cot_reasons(self, *args: Any, **kwargs: Any):
        """Raise because Jev cannot generate textual explanations."""
        raise NotImplementedError(
            "TypeSafe Jev is non-generative; use groundedness_measure instead."
        )
