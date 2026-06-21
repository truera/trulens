import pytest
import os
os.environ["TRULENS_OTEL_TRACING"] = "0"

from typing import Optional, Sequence, Dict, Type, ClassVar
from pydantic import BaseModel
from trulens.core.feedback import Feedback
from trulens.feedback.llm_provider import LLMProvider
from trulens.core.feedback.endpoint import Endpoint

class MockEndpoint(Endpoint):
    def run_in_pace(self, func, *args, **kwargs):
        return func(*args, **kwargs)

class MockLLMProvider(LLMProvider):
    model_config: ClassVar[dict] = {"extra": "allow"}
    
    model_engine: str = "mock-model"
    mock_response: str = ""
    
    def __init__(self, mock_response: str, **kwargs):
        super().__init__(endpoint=MockEndpoint(name="mock_endpoint"), mock_response=mock_response, **kwargs)

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> str:
        return self.mock_response

test_cases = [
    ("plain_number", "2", 0.2),
    ("with_explanation", "The relevance is moderate. Score: 2", 0.2),
    ("json_format", '{"criteria": "...", "supporting_evidence": "...", "score": 2}', 0.2),
    ("with_markdown", "```json\n{\\"score\\": 2}\n```", 0.2),
    ("whitespace_variations", "\n  Score: 2  \n", 0.2),
    ("float_format", "2.5", 0.25),
]

@pytest.mark.parametrize("name, response_text, expected_normalized_score", test_cases)
def test_score_parsing_pipeline(name, response_text, expected_normalized_score):
    provider = MockLLMProvider(mock_response=response_text)
    
    # Run through the relevance feedback method
    result = provider.relevance(prompt="What is the question?", response="This is the answer")
    
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
    assert result == expected_normalized_score
