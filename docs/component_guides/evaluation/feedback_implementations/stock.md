# Stock Feedback Functions

## Classification-based

### 🤗 HuggingFace

API Reference: [HuggingFace][trulens.providers.huggingface.Huggingface].

### OpenAI

API Reference: [OpenAI][trulens.providers.openai.OpenAI].

## Generation-based

### LLMProvider

API Reference: [LLMProvider][trulens.feedback.LLMProvider].

## Deterministic

No LLM calls and nothing sent to a provider.

### PII leakage

Scores text for personal data with a fixed format (SSN, card number, email,
phone, IPv4). It does not find names, addresses or medical details, so a score
of 1.0 means none of those formats were found.

```python
from trulens.core import Metric
from trulens.feedback.deterministic import pii_leakage

f_pii = Metric(implementation=pii_leakage, name="PII leakage").on_output()
```

API Reference: [pii_leakage][trulens.feedback.deterministic.pii_leakage].

## Embedding-based

### Embeddings

API Reference: [Embeddings][trulens.feedback.embeddings].

## Combinations

### Ground Truth Agreement

API Reference: [GroundTruthAgreement][trulens.feedback.groundtruth]
