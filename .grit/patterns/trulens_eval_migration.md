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
  `from trulens_eval import Tru` => `from trulens.core import TruSession`,
  `from trulens_eval import Select` => `from trulens.core import Select`,
  `from trulens_eval import Feedback` => `from trulens.core import Feedback`,
  `from trulens_eval import TruCustomApp` => `from trulens.apps.custom import TruCustomApp`,
  `from trulens_eval import TruChain` => `from trulens.apps.langchain import TruChain`,
  `from trulens_eval import TruLlama` => `from trulens.apps.llamaindex import TruLlama`,
  `from trulens_eval import OpenAI` => `from trulens.providers.openai import OpenAI`,
  `from trulens_eval import Huggingface` => `from trulens.providers.huggingface import Huggingface`,
  `from trulens_eval.guardrails.base import context_filter` => `from trulens.core.guardrails.base import context_filter`,
  `from trulens_eval.guardrails.langchain import WithFeedbackFilterDocuments` => `from trulens.apps.langchain.guardrails import WithFeedbackFilterDocuments`,
  `from trulens_eval.guardrails.llama import WithFeedbackFilterDocuments` => `from trulens.apps.llamaindex.guardrails import WithFeedbackFilterNodes`,
  `tru.run_dashboard($port)` => `from trulens.dashboard import run_dashboard; run_dashboard($port)`
}
