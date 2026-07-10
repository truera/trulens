# Agentic Evaluations

TruLens provides seven feedback functions specifically designed for evaluating agentic
systems. These functions assess different aspects of an agent's plan and execution,
giving you fine-grained insight into where your agent succeeds or struggles.

All agentic evaluators are available via any TruLens feedback provider (e.g., OpenAI,
Bedrock, LiteLLM, Huggingface) and operate on the agent's trace — the full record of
thinking, planning, and actions taken.

## Installation

```bash
pip install trulens trulens-providers-openai
```

## Quick start

```python
from trulens.core import TruSession, Metric, Selector
from trulens.apps.langchain import TruChain
from trulens.providers.openai import OpenAI

session = TruSession()
provider = OpenAI()

# Build a feedback function using the Metric API
f_logical_consistency = Metric(
    provider.logical_consistency_with_cot_reasons,
    name="Logical Consistency",
    selectors={"trace": Selector(trace_level=True)},
)
```

## Evaluators

### Logical Consistency

**What it measures:** Whether every action, claim, and transition in the agent's trace is
explicitly justified by prior context. Flags hallucinations, unsupported assertions, and
logical leaps.

**Use when:** You need to verify that your agent's reasoning is grounded and coherent —
especially important for high-stakes domains like finance, legal, or medical.

```python
from trulens.core import Metric, Selector
from trulens.providers.openai import OpenAI

provider = OpenAI()

f_logical_consistency = Metric(
    provider.logical_consistency_with_cot_reasons,
    name="Logical Consistency",
    selectors={"trace": Selector(trace_level=True)},
)
```

---

### Execution Efficiency

**What it measures:** Whether the agent executed its plan without unnecessary repetition,
backtracking, or wasted computation. Penalizes redundant tool calls, preventable retries,
and poorly ordered steps.

**Use when:** You want to reduce latency and cost by identifying agents that loop or
retry unnecessarily.

```python
from trulens.core import Metric, Selector
from trulens.providers.openai import OpenAI

provider = OpenAI()

f_execution_efficiency = Metric(
    provider.execution_efficiency_with_cot_reasons,
    name="Execution Efficiency",
    selectors={"trace": Selector(trace_level=True)},
)
```

---

### Plan Adherence

**What it measures:** Whether the agent's execution followed its plan step-by-step.
Penalizes skipped steps, unauthorized reordering, and undocumented deviations.

**Use when:** You need the agent to follow a structured workflow exactly — e.g., a
multi-step research or data-processing pipeline.

```python
from trulens.core import Metric, Selector
from trulens.providers.openai import OpenAI

provider = OpenAI()

f_plan_adherence = Metric(
    provider.plan_adherence_with_cot_reasons,
    name="Plan Adherence",
    selectors={"trace": Selector(trace_level=True)},
)
```

---

### Plan Quality

**What it measures:** The intrinsic quality of the agent's plan — before execution.
Evaluates whether the plan correctly uses available tools, addresses the user's query,
and is structured logically. Does **not** evaluate whether the plan was followed.

**Use when:** You want to evaluate the agent's planning capability separately from its
execution. Useful for debugging planning vs. execution failures.

```python
from trulens.core import Metric, Selector
from trulens.providers.openai import OpenAI

provider = OpenAI()

f_plan_quality = Metric(
    provider.plan_quality_with_cot_reasons,
    name="Plan Quality",
    selectors={"trace": Selector(trace_level=True)},
)
```

---

### Tool Selection

**What it measures:** Whether the agent chose the *right* tools for each subtask given
the available tool descriptions. Focuses purely on selection suitability — not how the
tool was called or how efficiently it was used.

**Use when:** You want to detect agents that consistently pick suboptimal or irrelevant
tools, or ignore mandated tools.

```python
from trulens.core import Metric, Selector
from trulens.providers.openai import OpenAI

provider = OpenAI()

f_tool_selection = Metric(
    provider.tool_selection_with_cot_reasons,
    name="Tool Selection",
    selectors={"trace": Selector(trace_level=True)},
)
```

---

### Tool Calling

**What it measures:** The quality of tool invocations — argument validity, semantic
appropriateness, precondition satisfaction, and how well the agent interprets tool
outputs. Does **not** judge which tool was chosen (see `Tool Selection`) or external
tool reliability (see `Tool Quality`).

**Use when:** You want to find agents that pass invalid arguments, misread tool outputs,
or fail to handle tool errors gracefully.

```python
from trulens.core import Metric, Selector
from trulens.providers.openai import OpenAI

provider = OpenAI()

f_tool_calling = Metric(
    provider.tool_calling_with_cot_reasons,
    name="Tool Calling",
    selectors={"trace": Selector(trace_level=True)},
)
```

---

### Tool Quality

**What it measures:** The reliability and output quality of the *tools themselves* as
observed in the trace — external errors (5xx, 429, 401), timeouts, flakiness, and
domain-specific output quality (e.g., search relevance). Independent of agent behavior.

**Use when:** You want to separate tool-side failures from agent-side failures. Useful
for infrastructure monitoring and vendor evaluation.

```python
from trulens.core import Metric, Selector
from trulens.providers.openai import OpenAI

provider = OpenAI()

f_tool_quality = Metric(
    provider.tool_quality_with_cot_reasons,
    name="Tool Quality",
    selectors={"trace": Selector(trace_level=True)},
)
```

---

## Using all seven evaluators together

```python
from trulens.core import TruSession, Metric, Selector
from trulens.providers.openai import OpenAI

session = TruSession()
provider = OpenAI()

feedbacks = [
    Metric(provider.logical_consistency_with_cot_reasons, name="Logical Consistency", selectors={"trace": Selector(trace_level=True)}),
    Metric(provider.execution_efficiency_with_cot_reasons, name="Execution Efficiency", selectors={"trace": Selector(trace_level=True)}),
    Metric(provider.plan_adherence_with_cot_reasons, name="Plan Adherence", selectors={"trace": Selector(trace_level=True)}),
    Metric(provider.plan_quality_with_cot_reasons, name="Plan Quality", selectors={"trace": Selector(trace_level=True)}),
    Metric(provider.tool_selection_with_cot_reasons, name="Tool Selection", selectors={"trace": Selector(trace_level=True)}),
    Metric(provider.tool_calling_with_cot_reasons, name="Tool Calling", selectors={"trace": Selector(trace_level=True)}),
    Metric(provider.tool_quality_with_cot_reasons, name="Tool Quality", selectors={"trace": Selector(trace_level=True)}),
]
```

## Evaluator scope boundaries

The seven evaluators are designed to be **orthogonal** — each measures a distinct
aspect of agent behavior with minimal overlap:

| Evaluator | Scope |
|-----------|-------|
| Logical Consistency | Reasoning coherence across the full trace |
| Execution Efficiency | Workflow sequencing and resource use |
| Plan Adherence | Execution fidelity to the stated plan |
| Plan Quality | Intrinsic plan quality (strategy, not outcome) |
| Tool Selection | Choice of tool per subtask |
| Tool Calling | Argument formation and output interpretation |
| Tool Quality | External tool/service reliability |

This separation lets you diagnose failures precisely: a low `Plan Adherence` score with
a high `Plan Quality` score means the agent planned well but failed to follow through.
A low `Tool Calling` score with a high `Tool Selection` score means the right tools were
chosen but invoked incorrectly.

## Related

- [Custom Feedback Functions](./feedback_implementations/custom_feedback_functions.ipynb)
- [Feedback Selectors](./feedback_selectors/index.md)
- [Agentic Evaluation Cookbook](../../../examples/cookbooks/agentic_evaluations.ipynb)
