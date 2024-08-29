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
  `from trulens_eval import TruBasicApp` => `from trulens.apps.basic import TruBasicApp`,
  `from trulens_eval.tru_custom_app import instrument` => `from trulens.apps.custom import instrument`,
  `from trulens_eval import TruChain` => `from trulens.apps.langchain import TruChain`,
  `from trulens_eval import TruLlama` => `from trulens.apps.llamaindex import TruLlama`,
  `from trulens_eval import TruRails` => `from trulens.apps.nemo import TruRails`,
  `from trulens_eval import OpenAI` => `from trulens.providers.openai import OpenAI`,
  `from trulens_eval import Huggingface` => `from trulens.providers.huggingface import Huggingface`,
  `from trulens_eval.guardrails.base import context_filter` => `from trulens.core.guardrails.base import context_filter`,
  `from trulens_eval.guardrails.langchain import WithFeedbackFilterDocuments` => `from trulens.apps.langchain.guardrails import WithFeedbackFilterDocuments`,
  `from trulens_eval.guardrails.llama import WithFeedbackFilterDocuments` => `from trulens.apps.llamaindex.guardrails import WithFeedbackFilterNodes`,
  `$var.run_dashboard($port)` => `from trulens.dashboard import run_dashboard; run_dashboard(session=$var, $port)`,
  `$var = Tru($connection)` => `$var = TruSession($connection)`,
  `$var.reset_database()` => `$var.reset_database()`,
  `from trulens_eval.utils.display import get_feedback_result` => `from trulens.dashboard.display import get_feedback_result`,
}
```

## Migrate and use TruSession

```python
from trulens_eval import Tru
tru = Tru()
tru.reset_database()
```
```python
from trulens.core import TruSession
from trulens.core.database.connector.default import DefaultDBConnector
connector = DefaultDBConnector()
session = TruSession(connector)
session.reset_database()
```

## Updates to Dashboard

```python
from trulens_eval import Tru
tru = Tru()
tru.run_dashboard()
```
```python
from trulens.core import TruSession
from trulens.core.database.connector.default import DefaultDBConnector
connector = DefaultDBConnector()
session = TruSession(connector)
from trulens.dashboard import run_dashboard
run_dashboard(session,)
```

## Updates to Dashboard with port

```python
from trulens_eval import Tru
tru = Tru()
tru.run_dashboard(port=888)
```
```python
from trulens.core import TruSession
from trulens.core.database.connector.default import DefaultDBConnector
connector = DefaultDBConnector()
session = TruSession(connector)
from trulens.dashboard import run_dashboard
run_dashboard(session, port=888)
```

## Migrations to TruLens Core

```python
from trulens_eval import Select
from trulens_eval import Feedback
from trulens_eval.guardrails.base import context_filter
```
```python
from trulens.core import Select
from trulens.core import Feedback
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
from trulens.apps.langchain import TruChain
from trulens.apps.langchain.guardrails import WithFeedbackFilterDocuments
```

## Updates to working with Llama-Index

```python
from trulens_eval import TruLlama
from trulens_eval.guardrails.llama import WithFeedbackFilterDocuments
```
```python
from trulens.apps.llamaindex import TruLlama
from trulens.apps.llamaindex.guardrails import WithFeedbackFilterNodes
```

## Updates to working with NeMo Guardrails

```python
from trulens_eval import TruRails
```
```python
from trulens.apps.nemo import TruRails
```
