---
title: Upgrade to TruLens v1.x
tags: [python, trulens, migration]
---

# Upgrade to TruLens v1.x

This pattern upgrades code to TruLens v1.x by replacing deprecated imports and class usage.

```grit
engine marzano(0.1)
language python

any {
  `Tru($session)` => `TruSession($session)`,
  `from trulens_eval.tru_custom_app import instrument` => `from trulens.apps.custom import instrument`,
  `from trulens_eval import TruRails` => `from trulens.apps.nemo import TruRails`,
  `from trulens_eval import OpenAI` => `from trulens.providers.openai import OpenAI`,
  `from trulens_eval import Huggingface` => `from trulens.providers.huggingface import Huggingface`,
  `from trulens_eval.guardrails.base import context_filter` => `from trulens.core.guardrails.base import context_filter`,
  `from trulens_eval.guardrails.langchain import WithFeedbackFilterDocuments` => `from trulens.apps.langchain.guardrails import WithFeedbackFilterDocuments`,
  `from trulens_eval.guardrails.llama import WithFeedbackFilterDocuments` => `from trulens.apps.llamaindex.guardrails import WithFeedbackFilterNodes`,
  `$var.run_dashboard($port)` => `from trulens.dashboard import run_dashboard; run_dashboard(session=$var, $port)`,
  `$var = Tru($connection)` => `from trulens.core.database.connector.default import DefaultDBConnector; connector = DefaultDBConnector($connection); $var = TruSession(connector=connector)`,
  `from trulens_eval.utils.display import get_feedback_result` => `from trulens.dashboard.display import get_feedback_result`,
    `from trulens_eval import TruBasicApp` => `from trulens.apps.basic import TruBasicApp`,
  `from trulens_eval import TruBasicApp as $alias` => `from trulens.apps.basic import TruBasicApp as $alias`,
  `from trulens_eval import TruCustomApp` => `from trulens.apps.custom import TruCustomApp`,
  `from trulens_eval import TruCustomApp as $alias` => `from trulens.apps.custom import TruCustomApp as $alias`,
  `from trulens_eval import TruVirtual` => `from trulens.apps.virtual import TruVirtual`,
  `from trulens_eval import TruVirtual as $alias` => `from trulens.apps.virtual import TruVirtual as $alias`,
  `from trulens_eval import Feedback` => `from trulens.core.feedback.feedback import Feedback`,
  `from trulens_eval import Feedback as $alias` => `from trulens.core.feedback.feedback import Feedback as $alias`,
  `from trulens_eval import Provider` => `from trulens.core.feedback.provider import Provider`,
  `from trulens_eval import Provider as $alias` => `from trulens.core.feedback.provider import Provider as $alias`,
  `from trulens_eval import Select` => `from trulens.core.schema import Select`,
  `from trulens_eval import Select as $alias` => `from trulens.core.schema import Select as $alias`,
  `from trulens_eval import FeedbackMode` => `from trulens.core.schema.feedback import FeedbackMode`,
  `from trulens_eval import FeedbackMode as $alias` => `from trulens.core.schema.feedback import FeedbackMode as $alias`,
  `from trulens_eval import Tru` => `from trulens.core.session import TruSession`,
  `from trulens_eval import Tru as $alias` => `from trulens.core.session import TruSession as $alias`,
  `from trulens_eval import TP` => `from trulens.core.utils.threading import TP`,
  `from trulens_eval import TP as $alias` => `from trulens.core.utils.threading import TP as $alias`,
  `from trulens_eval import LiteLLM` => `from trulens.providers.litellm.provider import LiteLLM`,
  `from trulens_eval import LiteLLM as $alias` => `from trulens.providers.litellm.provider import LiteLLM as $alias`,
  `from trulens_eval import Bedrock` => `from trulens.providers.bedrock.provider import Bedrock`,
  `from trulens_eval import Bedrock as $alias` => `from trulens.providers.bedrock.provider import Bedrock as $alias`,
  `from trulens_eval import AzureOpenAI` => `from trulens.providers.openai.provider import AzureOpenAI`,
  `from trulens_eval import AzureOpenAI as $alias` => `from trulens.providers.openai.provider import AzureOpenAI as $alias`,
  `from trulens_eval import OpenAI` => `from trulens.providers.openai.provider import OpenAI`,
  `from trulens_eval import OpenAI as $alias` => `from trulens.providers.openai.provider import OpenAI as $alias`,
  `from trulens_eval import Cortex` => `from trulens.providers.cortex.provider import Cortex`,
  `from trulens_eval import Cortex as $alias` => `from trulens.providers.cortex.provider import Cortex as $alias`,
  `from trulens_eval import Huggingface` => `from trulens.providers.huggingface.provider import Huggingface`,
  `from trulens_eval import Huggingface as $alias` => `from trulens.providers.huggingface.provider import Huggingface as $alias`,
  `from trulens_eval import HuggingfaceLocal` => `from trulens.providers.huggingface.provider import HuggingfaceLocal`,
  `from trulens_eval import HuggingfaceLocal as $alias` => `from trulens.providers.huggingface.provider import HuggingfaceLocal as $alias`,
  `from trulens_eval import Langchain` => `from trulens.providers.langchain.provider import Langchain`,
  `from trulens_eval import Langchain as $alias` => `from trulens.providers.langchain.provider import Langchain as $alias`,
  `from trulens_eval import TruChain` => `from trulens.apps.langchain.tru_chain import TruChain`,
  `from trulens_eval import TruChain as $alias` => `from trulens.apps.langchain.tru_chain import TruChain as $alias`,
  `from trulens_eval import TruLlama` => `from trulens.apps.llamaindex.tru_llama import TruLlama`,
  `from trulens_eval import TruLlama as $alias` => `from trulens.apps.llamaindex.tru_llama import TruLlama as $alias`,
  `from trulens_eval import TruRails` => `from trulens.apps.nemo.tru_rails import TruRails`,
  `from trulens_eval import TruRails as $alias` => `from trulens.apps.nemo.tru_rails import TruRails as $alias`,
  `from trulens_eval.feedback import Embeddings` => `from trulens.feedback.embeddings import Embeddings`,
  `from trulens_eval.feedback import Embeddings as $alias` => `from trulens.feedback.embeddings import Embeddings as $alias`,
  `from trulens_eval.feedback import Feedback` => `from trulens.feedback.feedback import Feedback`,
  `from trulens_eval.feedback import Feedback as $alias` => `from trulens.feedback.feedback import Feedback as $alias`,
  `from trulens_eval.feedback import GroundTruthAgreement` => `from trulens.feedback.groundtruth import GroundTruthAgreement`,
  `from trulens_eval.feedback import GroundTruthAgreement as $alias` => `from trulens.feedback.groundtruth import GroundTruthAgreement as $alias`,
  `from trulens_eval.feedback import LiteLLM` => `from trulens.providers.litellm.provider import LiteLLM`,
  `from trulens_eval.feedback import LiteLLM as $alias` => `from trulens.providers.litellm.provider import LiteLLM as $alias`,
  `from trulens_eval.feedback import Bedrock` => `from trulens.providers.bedrock.provider import Bedrock`,
  `from trulens_eval.feedback import Bedrock as $alias` => `from trulens.providers.bedrock.provider import Bedrock as $alias`,
  `from trulens_eval.feedback import AzureOpenAI` => `from trulens.providers.openai.provider import AzureOpenAI`,
  `from trulens_eval.feedback import AzureOpenAI as $alias` => `from trulens.providers.openai.provider import AzureOpenAI as $alias`,
  `from trulens_eval.feedback import OpenAI` => `from trulens.providers.openai.provider import OpenAI`,
  `from trulens_eval.feedback import OpenAI as $alias` => `from trulens.providers.openai.provider import OpenAI as $alias`,
  `from trulens_eval.feedback import Huggingface` => `from trulens.providers.huggingface.provider import Huggingface`,
  `from trulens_eval.feedback import Huggingface as $alias` => `from trulens.providers.huggingface.provider import Huggingface as $alias`,
  `from trulens_eval.feedback import HuggingfaceLocal` => `from trulens.providers.huggingface.provider import HuggingfaceLocal`,
  `from trulens_eval.feedback import HuggingfaceLocal as $alias` => `from trulens.providers.huggingface.provider import HuggingfaceLocal as $alias`,
  `from trulens_eval.feedback import Langchain` => `from trulens.providers.langchain.provider import Langchain`,
  `from trulens_eval.feedback import Langchain as $alias` => `from trulens.providers.langchain.provider import Langchain as $alias`,
  `from trulens_eval.feedback import Cortex` => `from trulens.providers.cortex.provider import Cortex`,
  `from trulens_eval.feedback import Cortex as $alias` => `from trulens.providers.cortex.provider import Cortex as $alias`,
  `from trulens_eval.feedback.embeddings import Embeddings` => `from trulens.feedback.embeddings import Embeddings`,
  `from trulens_eval.feedback.embeddings import Embeddings as $alias` => `from trulens.feedback.embeddings import Embeddings as $alias`,
  `from trulens_eval.feedback.feedback import InvalidSelector` => `from trulens.core.feedback.feedback import InvalidSelector`,
  `from trulens_eval.feedback.feedback import InvalidSelector as $alias` => `from trulens.core.feedback.feedback import InvalidSelector as $alias`,
  `from trulens_eval.feedback.feedback import SkipEval` => `from trulens.core.feedback.feedback import SkipEval`,
  `from trulens_eval.feedback.feedback import SkipEval as $alias` => `from trulens.core.feedback.feedback import SkipEval as $alias`,
  `from trulens_eval.feedback.feedback import Feedback` => `from trulens.feedback.feedback import Feedback`,
  `from trulens_eval.feedback.feedback import Feedback as $alias` => `from trulens.feedback.feedback import Feedback as $alias`,
  `from trulens_eval.feedback.feedback import rag_triad` => `from trulens.feedback.feedback import rag_triad`,
  `from trulens_eval.feedback.feedback import rag_triad as $alias` => `from trulens.feedback.feedback import rag_triad as $alias`,
  `from trulens_eval.feedback.groundtruth import GroundTruthAgreement` => `from trulens.feedback.groundtruth import GroundTruthAgreement`,
  `from trulens_eval.feedback.groundtruth import GroundTruthAgreement as $alias` => `from trulens.feedback.groundtruth import GroundTruthAgreement as $alias`,
    `from trulens_eval.feedback.provider import Provider` => `from trulens.core.feedback.provider import Provider`,
  `from trulens_eval.feedback.provider import Provider as $alias` => `from trulens.core.feedback.provider import Provider as $alias`,
  `from trulens_eval.feedback.provider import imports_utils` => `from trulens.core.utils import imports`,
  `from trulens_eval.feedback.provider import imports_utils as $alias` => `from trulens.core.utils import imports as $alias`,
  `from trulens_eval.feedback.provider import LiteLLM` => `from trulens.providers.litellm.provider import LiteLLM`,
  `from trulens_eval.feedback.provider import LiteLLM as $alias` => `from trulens.providers.litellm.provider import LiteLLM as $alias`,
  `from trulens_eval.feedback.provider import Bedrock` => `from trulens.providers.bedrock.provider import Bedrock`,
  `from trulens_eval.feedback.provider import Bedrock as $alias` => `from trulens.providers.bedrock.provider import Bedrock as $alias`,
  `from trulens_eval.feedback.provider import AzureOpenAI` => `from trulens.providers.openai.provider import AzureOpenAI`,
  `from trulens_eval.feedback.provider import AzureOpenAI as $alias` => `from trulens.providers.openai.provider import AzureOpenAI as $alias`,
  `from trulens_eval.feedback.provider import OpenAI` => `from trulens.providers.openai.provider import OpenAI`,
  `from trulens_eval.feedback.provider import OpenAI as $alias` => `from trulens.providers.openai.provider import OpenAI as $alias`,
  `from trulens_eval.feedback.provider import Cortex` => `from trulens.providers.cortex.provider import Cortex`,
  `from trulens_eval.feedback.provider import Cortex as $alias` => `from trulens.providers.cortex.provider import Cortex as $alias`,
  `from trulens_eval.feedback.provider import Huggingface` => `from trulens.providers.huggingface.provider import Huggingface`,
  `from trulens_eval.feedback.provider import Huggingface as $alias` => `from trulens.providers.huggingface.provider import Huggingface as $alias`,
  `from trulens_eval.feedback.provider import HuggingfaceLocal` => `from trulens.providers.huggingface.provider import HuggingfaceLocal`,
  `from trulens_eval.feedback.provider import HuggingfaceLocal as $alias` => `from trulens.providers.huggingface.provider import HuggingfaceLocal as $alias`,
  `from trulens_eval.feedback.provider import Langchain` => `from trulens.providers.langchain.provider import Langchain`,
  `from trulens_eval.feedback.provider import Langchain as $alias` => `from trulens.providers.langchain.provider import Langchain as $alias`,
  `from trulens_eval.feedback.provider.base import Provider` => `from trulens.core.feedback.provider import Provider`,
  `from trulens_eval.feedback.provider.base import Provider as $alias` => `from trulens.core.feedback.provider import Provider as $alias`,
  `from trulens_eval.feedback.provider.base import LLMProvider` => `from trulens.feedback.llm_provider import LLMProvider`,
  `from trulens_eval.feedback.provider.base import LLMProvider as $alias` => `from trulens.feedback.llm_provider import LLMProvider as $alias`,
  `from trulens_eval.feedback.provider.bedrock import Bedrock` => `from trulens.providers.bedrock.provider import Bedrock`,
  `from trulens_eval.feedback.provider.bedrock import Bedrock as $alias` => `from trulens.providers.bedrock.provider import Bedrock as $alias`,
  `from trulens_eval.feedback.provider.cortex import Cortex` => `from trulens.providers.cortex.provider import Cortex`,
  `from trulens_eval.feedback.provider.cortex import Cortex as $alias` => `from trulens.providers.cortex.provider import Cortex as $alias`,
  `from trulens_eval.feedback.provider.endpoint import Endpoint` => `from trulens.core.feedback.endpoint import Endpoint`,
  `from trulens_eval.feedback.provider.endpoint import Endpoint as $alias` => `from trulens.core.feedback.endpoint import Endpoint as $alias`,
  `from trulens_eval.feedback.provider.endpoint import imports_utils` => `from trulens.core.utils import imports`,
  `from trulens_eval.feedback.provider.endpoint import imports_utils as $alias` => `from trulens.core.utils import imports as $alias`,
  `from trulens_eval.feedback.provider.endpoint import DummyEndpoint` => `from trulens.feedback.dummy.endpoint import DummyEndpoint`,
  `from trulens_eval.feedback.provider.endpoint import DummyEndpoint as $alias` => `from trulens.feedback.dummy.endpoint import DummyEndpoint as $alias`,
  `from trulens_eval.feedback.provider.endpoint import LiteLLMEndpoint` => `from trulens.providers.litellm.endpoint import LiteLLMEndpoint`,
  `from trulens_eval.feedback.provider.endpoint import LiteLLMEndpoint as $alias` => `from trulens.providers.litellm.endpoint import LiteLLMEndpoint as $alias`,
  `from trulens_eval.feedback.provider.endpoint import BedrockEndpoint` => `from trulens.providers.bedrock.endpoint import BedrockEndpoint`,
  `from trulens_eval.feedback.provider.endpoint import BedrockEndpoint as $alias` => `from trulens.providers.bedrock.endpoint import BedrockEndpoint as $alias`,
  `from trulens_eval.feedback.provider.endpoint import OpenAIClient` => `from trulens.providers.openai.endpoint import OpenAIClient`,
  `from trulens_eval.feedback.provider.endpoint import OpenAIClient as $alias` => `from trulens.providers.openai.endpoint import OpenAIClient as $alias`,
  `from trulens_eval.feedback.provider.endpoint import OpenAIEndpoint` => `from trulens.providers.openai.endpoint import OpenAIEndpoint`,
  `from trulens_eval.feedback.provider.endpoint import OpenAIEndpoint as $alias` => `from trulens.providers.openai.endpoint import OpenAIEndpoint as $alias`,
  `from trulens_eval.feedback.provider.endpoint import CortexEndpoint` => `from trulens.providers.cortex.endpoint import CortexEndpoint`,
  `from trulens_eval.feedback.provider.endpoint import CortexEndpoint as $alias` => `from trulens.providers.cortex.endpoint import CortexEndpoint as $alias`,
  `from trulens_eval.feedback.provider.endpoint import LangchainEndpoint` => `from trulens.providers.langchain.endpoint import LangchainEndpoint`,
  `from trulens_eval.feedback.provider.endpoint import LangchainEndpoint as $alias` => `from trulens.providers.langchain.endpoint import LangchainEndpoint as $alias`,
  `from trulens_eval.feedback.provider.endpoint import HuggingfaceEndpoint` => `from trulens.providers.huggingface.endpoint import HuggingfaceEndpoint`,
  `from trulens_eval.feedback.provider.endpoint import HuggingfaceEndpoint as $alias` => `from trulens.providers.huggingface.endpoint import HuggingfaceEndpoint as $alias`,
  `from trulens_eval.feedback.provider.endpoint.base import DEFAULT_NETWORK_TIMEOUT` => `from trulens.core.feedback.endpoint import DEFAULT_NETWORK_TIMEOUT`,
  `from trulens_eval.feedback.provider.endpoint.base import DEFAULT_NETWORK_TIMEOUT as $alias` => `from trulens.core.feedback.endpoint import DEFAULT_NETWORK_TIMEOUT as $alias`,
  `from trulens_eval.feedback.provider.endpoint.base import DEFAULT_RPM` => `from trulens.core.feedback.endpoint import DEFAULT_RPM`,
  `from trulens_eval.feedback.provider.endpoint.base import DEFAULT_RPM as $alias` => `from trulens.core.feedback.endpoint import DEFAULT_RPM as $alias`,
  `from trulens_eval.feedback.provider.endpoint.base import INSTRUMENT` => `from trulens.core.feedback.endpoint import INSTRUMENT`,
  `from trulens_eval.feedback.provider.endpoint.base import INSTRUMENT as $alias` => `from trulens.core.feedback.endpoint import INSTRUMENT as $alias`,
  `from trulens_eval.feedback.provider.endpoint.base import Endpoint` => `from trulens.core.feedback.endpoint import Endpoint`,
  `from trulens_eval.feedback.provider.endpoint.base import Endpoint as $alias` => `from trulens.core.feedback.endpoint import Endpoint as $alias`,
  `from trulens_eval.feedback.provider.endpoint.base import EndpointCallback` => `from trulens.core.feedback.endpoint import EndpointCallback`,
  `from trulens_eval.feedback.provider.endpoint.base import EndpointCallback as $alias` => `from trulens.core.feedback.endpoint import EndpointCallback as $alias`,
  `from trulens_eval.feedback.provider.endpoint.base import DummyEndpoint` => `from trulens.feedback.dummy.endpoint import DummyEndpoint`,
  `from trulens_eval.feedback.provider.endpoint.base import DummyEndpoint as $alias` => `from trulens.feedback.dummy.endpoint import DummyEndpoint as $alias`,
  `from trulens_eval.feedback.provider.endpoint.bedrock import INSTRUMENT` => `from trulens.providers.bedrock.endpoint import INSTRUMENT`,
  `from trulens_eval.feedback.provider.endpoint.bedrock import INSTRUMENT as $alias` => `from trulens.providers.bedrock.endpoint import INSTRUMENT as $alias`,
  `from trulens_eval.feedback.provider.endpoint.bedrock import BedrockCallback` => `from trulens.providers.bedrock.endpoint import BedrockCallback`,
  `from trulens_eval.feedback.provider.endpoint.bedrock import BedrockCallback as $alias` => `from trulens.providers.bedrock.endpoint import BedrockCallback as $alias`,
  `from trulens_eval.feedback.provider.endpoint.bedrock import BedrockEndpoint` => `from trulens.providers.bedrock.endpoint import BedrockEndpoint`,
  `from trulens_eval.feedback.provider.endpoint.bedrock import BedrockEndpoint as $alias` => `from trulens.providers.bedrock.endpoint import BedrockEndpoint as $alias`,
  `from trulens_eval.feedback.provider.endpoint.cortex import CortexCallback` => `from trulens.providers.cortex.endpoint import CortexCallback`,
  `from trulens_eval.feedback.provider.endpoint.cortex import CortexCallback as $alias` => `from trulens.providers.cortex.endpoint import CortexCallback as $alias`,
  `from trulens_eval.feedback.provider.endpoint.cortex import CortexEndpoint` => `from trulens.providers.cortex.endpoint import CortexEndpoint`,
  `from trulens_eval.feedback.provider.endpoint.cortex import CortexEndpoint as $alias` => `from trulens.providers.cortex.endpoint import CortexEndpoint as $alias`,
  `from trulens_eval.feedback.provider.endpoint.hugs import HuggingfaceCallback` => `from trulens.providers.huggingface.endpoint import HuggingfaceCallback`,
  `from trulens_eval.feedback.provider.endpoint.hugs import HuggingfaceCallback as $alias` => `from trulens.providers.huggingface.endpoint import HuggingfaceCallback as $alias`,
  `from trulens_eval.feedback.provider.endpoint.hugs import HuggingfaceEndpoint` => `from trulens.providers.huggingface.endpoint import HuggingfaceEndpoint`,
  `from trulens_eval.feedback.provider.endpoint.hugs import HuggingfaceEndpoint as $alias` => `from trulens.providers.huggingface.endpoint import HuggingfaceEndpoint as $alias`,
  `from trulens_eval.feedback.provider.endpoint.langchain import LangchainCallback` => `from trulens.providers.langchain.endpoint import LangchainCallback`,
  `from trulens_eval.feedback.provider.endpoint.langchain import LangchainCallback as $alias` => `from trulens.providers.langchain.endpoint import LangchainCallback as $alias`,
  `from trulens_eval.feedback.provider.endpoint.langchain import LangchainEndpoint` => `from trulens.providers.langchain.endpoint import LangchainEndpoint`,
  `from trulens_eval.feedback.provider.endpoint.langchain import LangchainEndpoint as $alias` => `from trulens.providers.langchain.endpoint import LangchainEndpoint as $alias`,
  `from trulens_eval.feedback.provider.endpoint.litellm import LiteLLMCallback` => `from trulens.providers.litellm.endpoint import LiteLLMCallback`,
  `from trulens_eval.feedback.provider.endpoint.litellm import LiteLLMCallback as $alias` => `from trulens.providers.litellm.endpoint import LiteLLMCallback as $alias`,
  `from trulens_eval.feedback.provider.endpoint.litellm import LiteLLMEndpoint` => `from trulens.providers.litellm.endpoint import LiteLLMEndpoint`,
  `from trulens_eval.feedback.provider.endpoint.litellm import LiteLLMEndpoint as $alias` => `from trulens.providers.litellm.endpoint import LiteLLMEndpoint as $alias`,
  `from trulens_eval.feedback.provider.endpoint.openai import CLASS_INFO` => `from trulens.providers.openai.endpoint import CLASS_INFO`,
  `from trulens_eval.feedback.provider.endpoint.openai import CLASS_INFO as $alias` => `from trulens.providers.openai.endpoint import CLASS_INFO as $alias`,
  `from trulens_eval.feedback.provider.endpoint.openai import OpenAICallback` => `from trulens.providers.openai.endpoint import OpenAICallback`,
  `from trulens_eval.feedback.provider.endpoint.openai import OpenAICallback as $alias` => `from trulens.providers.openai.endpoint import OpenAICallback as $alias`,
  `from trulens_eval.feedback.provider.endpoint.openai import OpenAIClient` => `from trulens.providers.openai.endpoint import OpenAIClient`,
  `from trulens_eval.feedback.provider.endpoint.openai import OpenAIClient as $alias` => `from trulens.providers.openai.endpoint import OpenAIClient as $alias`,
  `from trulens_eval.feedback.provider.endpoint.openai import OpenAIEndpoint` => `from trulens.providers.openai.endpoint import OpenAIEndpoint`,
  `from trulens_eval.feedback.provider.endpoint.openai import OpenAIEndpoint as $alias` => `from trulens.providers.openai.endpoint import OpenAIEndpoint as $alias`,
  `from trulens_eval.feedback.provider.hugs import HUGS_CONTEXT_RELEVANCE_API_URL` => `from trulens.providers.huggingface.provider import HUGS_CONTEXT_RELEVANCE_API_URL`,
  `from trulens_eval.feedback.provider.hugs import HUGS_CONTEXT_RELEVANCE_API_URL as $alias` => `from trulens.providers.huggingface.provider import HUGS_CONTEXT_RELEVANCE_API_URL as $alias`,
  `from trulens_eval.feedback.provider.hugs import HUGS_DOCNLI_API_URL` => `from trulens.providers.huggingface.provider import HUGS_DOCNLI_API_URL`,
  `from trulens_eval.feedback.provider.hugs import HUGS_DOCNLI_API_URL as $alias` => `from trulens.providers.huggingface.provider import HUGS_DOCNLI_API_URL as $alias`,
  `from trulens_eval.feedback.provider.hugs import HUGS_DOCNLI_MODEL_PATH` => `from trulens.providers.huggingface.provider import HUGS_DOCNLI_MODEL_PATH`,
  `from trulens_eval.feedback.provider.hugs import HUGS_DOCNLI_MODEL_PATH as $alias` => `from trulens.providers.huggingface.provider import HUGS_DOCNLI_MODEL_PATH as $alias`,
  `from trulens_eval.feedback.provider.hugs import HUGS_HALLUCINATION_API_URL` => `from trulens.providers.huggingface.provider import HUGS_HALLUCINATION_API_URL`,
  `from trulens_eval.feedback.provider.hugs import HUGS_HALLUCINATION_API_URL as $alias` => `from trulens.providers.huggingface.provider import HUGS_HALLUCINATION_API_URL as $alias`,
  `from trulens_eval.feedback.provider.hugs import HUGS_LANGUAGE_API_URL` => `from trulens.providers.huggingface.provider import HUGS_LANGUAGE_API_URL`,
  `from trulens_eval.feedback.provider.hugs import HUGS_LANGUAGE_API_URL as $alias` => `from trulens.providers.huggingface.provider import HUGS_LANGUAGE_API_URL as $alias`,
  `from trulens_eval.feedback.provider.hugs import HUGS_NLI_API_URL` => `from trulens.providers.huggingface.provider import HUGS_NLI_API_URL`,
  `from trulens_eval.feedback.provider.hugs import HUGS_NLI_API_URL as $alias` => `from trulens.providers.huggingface.provider import HUGS_NLI_API_URL as $alias`,
  `from trulens_eval.feedback.provider.hugs import HUGS_PII_DETECTION_API_URL` => `from trulens.providers.huggingface.provider import HUGS_PII_DETECTION_API_URL`,
  `from trulens_eval.feedback.provider.hugs import HUGS_PII_DETECTION_API_URL as $alias` => `from trulens.providers.huggingface.provider import HUGS_PII_DETECTION_API_URL as $alias`,
  `from trulens_eval.feedback.provider.hugs import HUGS_SENTIMENT_API_URL` => `from trulens.providers.huggingface.provider import HUGS_SENTIMENT_API_URL`,
  `from trulens_eval.feedback.provider.hugs import HUGS_SENTIMENT_API_URL as $alias` => `from trulens.providers.huggingface.provider import HUGS_SENTIMENT_API_URL as $alias`,
  `from trulens_eval.feedback.provider.hugs import HUGS_TOXIC_API_URL` => `from trulens.providers.huggingface.provider import HUGS_TOXIC_API_URL`,
  `from trulens_eval.feedback.provider.hugs import HUGS_TOXIC_API_URL as $alias` => `from trulens.providers.huggingface.provider import HUGS_TOXIC_API_URL as $alias`,
  `from trulens_eval.feedback.provider.hugs import Dummy` => `from trulens.providers.huggingface.provider import Dummy`,
  `from trulens_eval.feedback.provider.hugs import Dummy as $alias` => `from trulens.providers.huggingface.provider import Dummy as $alias`,
  `from trulens_eval.feedback.provider.hugs import Huggingface` => `from trulens.providers.huggingface.provider import Huggingface`,
  `from trulens_eval.feedback.provider.hugs import Huggingface as $alias` => `from trulens.providers.huggingface.provider import Huggingface as $alias`,
  `from trulens_eval.feedback.provider.hugs import HuggingfaceBase` => `from trulens.providers.huggingface.provider import HuggingfaceBase`,
  `from trulens_eval.feedback.provider.hugs import HuggingfaceBase as $alias` => `from trulens.providers.huggingface.provider import HuggingfaceBase as $alias`,
  `from trulens_eval.feedback.provider.hugs import HuggingfaceLocal` => `from trulens.providers.huggingface.provider import HuggingfaceLocal`,
  `from trulens_eval.feedback.provider.hugs import HuggingfaceLocal as $alias` => `from trulens.providers.huggingface.provider import HuggingfaceLocal as $alias`,
  `from trulens_eval.feedback.provider.langchain import Langchain` => `from trulens.providers.langchain.provider import Langchain`,
  `from trulens_eval.feedback.provider.langchain import Langchain as $alias` => `from trulens.providers.langchain.provider import Langchain as $alias`,
  `from trulens_eval.feedback.provider.litellm import LiteLLM` => `from trulens.providers.litellm.provider import LiteLLM`,
  `from trulens_eval.feedback.provider.litellm import LiteLLM as $alias` => `from trulens.providers.litellm.provider import LiteLLM as $alias`,
  `from trulens_eval.feedback.provider.openai import CLASS_INFO` => `from trulens.providers.openai.provider import CLASS_INFO`,
  `from trulens_eval.feedback.provider.openai import CLASS_INFO as $alias` => `from trulens.providers.openai.provider import CLASS_INFO as $alias`,
  `from trulens_eval.feedback.provider.openai import AzureOpenAI` => `from trulens.providers.openai.provider import AzureOpenAI`,
  `from trulens_eval.feedback.provider.openai import AzureOpenAI as $alias` => `from trulens.providers.openai.provider import AzureOpenAI as $alias`,
  `from trulens_eval.feedback.provider.openai import OpenAI` => `from trulens.providers.openai.provider import OpenAI`,
  `from trulens_eval.feedback.provider.openai import OpenAI as $alias` => `from trulens.providers.openai.provider import OpenAI as $alias`,
  `from trulens_eval.generate_test_set import mod_generate` => `from trulens.benchmark.generate import generate_test_set`,
  `from trulens_eval.generate_test_set import mod_generate as $alias` => `from trulens.benchmark.generate import generate_test_set as $alias`,
  `from trulens_eval.guardrails.base import context_filter` => `from trulens.core.guardrails.base import context_filter`,
  `from trulens_eval.guardrails.base import context_filter as $alias` => `from trulens.core.guardrails.base import context_filter as $alias`,
  `from trulens_eval.guardrails.langchain import WithFeedbackFilterDocuments` => `from trulens.apps.langchain.guardrails import WithFeedbackFilterDocuments`,
  `from trulens_eval.guardrails.langchain import WithFeedbackFilterDocuments as $alias` => `from trulens.apps.langchain.guardrails import WithFeedbackFilterDocuments as $alias`,
  `from trulens_eval.guardrails.llama import WithFeedbackFilterNodes` => `from trulens.apps.llamaindex.guardrails import WithFeedbackFilterNodes`,
  `from trulens_eval.guardrails.llama import WithFeedbackFilterNodes as $alias` => `from trulens.apps.llamaindex.guardrails import WithFeedbackFilterNodes as $alias`,
  `from trulens_eval.instruments import AddInstruments` => `from trulens.core.instruments import AddInstruments`,
  `from trulens_eval.instruments import AddInstruments as $alias` => `from trulens.core.instruments import AddInstruments as $alias`,
  `from trulens_eval.instruments import Instrument` => `from trulens.core.instruments import Instrument`,
  `from trulens_eval.instruments import Instrument as $alias` => `from trulens.core.instruments import Instrument as $alias`,
  `from trulens_eval.instruments import WithInstrumentCallbacks` => `from trulens.core.instruments import WithInstrumentCallbacks`,
  `from trulens_eval.instruments import WithInstrumentCallbacks as $alias` => `from trulens.core.instruments import WithInstrumentCallbacks as $alias`,
  `from trulens_eval.instruments import class_filter_disjunction` => `from trulens.core.instruments import class_filter_disjunction`,
  `from trulens_eval.instruments import class_filter_disjunction as $alias` => `from trulens.core.instruments import class_filter_disjunction as $alias`,
  `from trulens_eval.instruments import class_filter_matches` => `from trulens.core.instruments import class_filter_matches`,
  `from trulens_eval.instruments import class_filter_matches as $alias` => `from trulens.core.instruments import class_filter_matches as $alias`,
  `from trulens_eval.instruments import instrument` => `from trulens.core.instruments import instrument`,
  `from trulens_eval.instruments import instrument as $alias` => `from trulens.core.instruments import instrument as $alias`,

}
```

## Migrate and use TruSession

```python
from trulens_eval import Tru
tru = Tru(database_url)
tru.reset_database()
```
```python
from trulens.core.session import TruSession
from trulens.core.database.connector.default import DefaultDBConnector
connector = DefaultDBConnector(database_url)
tru = TruSession(connector=connector)
tru.reset_database()
```

## Updates to Dashboard

```python
from trulens_eval import Tru
tru = Tru()
tru.run_dashboard()
```
```python
from trulens.core.session import TruSession
from trulens.core.database.connector.default import DefaultDBConnector
connector = DefaultDBConnector()
tru = TruSession(connector=connector)
from trulens.dashboard import run_dashboard
run_dashboard(session=tru,)
```

## Updates to Dashboard with port

```python
from trulens_eval import Tru
tru = Tru()
tru.run_dashboard(port=888)
```
```python
from trulens.core.session import TruSession
from trulens.core.database.connector.default import DefaultDBConnector
connector = DefaultDBConnector()
tru = TruSession(connector=connector)
from trulens.dashboard import run_dashboard
run_dashboard(session=tru, port=888)
```

## Migrations to TruLens Core

```python
from trulens_eval import Select
from trulens_eval import Feedback
from trulens_eval.guardrails.base import context_filter
```
```python
from trulens.core.schema import Select
from trulens.core.feedback.feedback import Feedback
from trulens.core.guardrails.base import context_filter
```

## Migrations to TruLens Apps

```python
from trulens_eval import TruCustomApp
from trulens_eval import TruBasicApp
```
```python
from trulens.apps.custom import TruCustomApp
from trulens.apps.basic import TruBasicApp
```

## Updates to working with Langchain

```python
from trulens_eval import TruChain
from trulens_eval.guardrails.langchain import WithFeedbackFilterDocuments
```
```python
from trulens.apps.langchain.tru_chain import TruChain
from trulens.apps.langchain.guardrails import WithFeedbackFilterDocuments
```

## Updates to working with Llama-Index

```python
from trulens_eval import TruLlama
from trulens_eval.guardrails.llama import WithFeedbackFilterDocuments
```
```python
from trulens.apps.llamaindex.tru_llama import TruLlama
from trulens.apps.llamaindex.guardrails import WithFeedbackFilterNodes
```

## Updates to working with NeMo Guardrails
```python
from trulens_eval import TruRails
```
```python
from trulens.apps.nemo.tru_rails import TruRails
```

## Using OpenAI Provider
```python
from trulens_eval import OpenAI
```
```python
from trulens.providers.openai.provider import OpenAI
```

## Using OpenAI Provider Full Import Path
```python
from trulens_eval.feedback.provider.openai import OpenAI
```
```python
from trulens.providers.openai.provider import OpenAI
```

## Using AzureOpenAI Provider
```python
from trulens_eval import AzureOpenAI
```
```python
from trulens.providers.openai.provider import AzureOpenAI
```

## Using AzureOpenAI Provider Full Import Path
```python
from trulens_eval.feedback.provider.openai import AzureOpenAI
```
```python
from trulens.providers.openai.provider import AzureOpenAI
```

## Using LiteLLM Provider
```python
from trulens_eval import LiteLLM
```
```python
from trulens.providers.litellm.provider import LiteLLM
```

## Using LiteLLM Provider Full Import Path
```python
from trulens_eval.feedback.provider.litellm import LiteLLM
```
```python
from trulens.providers.litellm.provider import LiteLLM
```

## Using Bedrock Provider
```python
from trulens_eval import Bedrock
```
```python
from trulens.providers.bedrock.provider import Bedrock
```

## Using Bedrock Provider Full Import Path
```python
from trulens_eval.feedback.provider.bedrock import Bedrock
```
```python
from trulens.providers.bedrock.provider import Bedrock
```

## Using Cortex Provider
```python
from trulens_eval import Cortex
```
```python
from trulens.providers.cortex.provider import Cortex
```

## Using Cortex Provider Full Import Path
```python
from trulens_eval.feedback.provider.cortex import Cortex
```
```python
from trulens.providers.cortex.provider import Cortex
```

## Using Langchain Provider
```python
from trulens_eval import Cortex
```
```python
from trulens.providers.cortex.provider import Cortex
```

## Using Langchain Provider Full Import Path
```python
from trulens_eval.feedback.provider.langchain import Langchain
```
```python
from trulens.providers.langchain.provider import Langchain
```
