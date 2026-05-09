# Agentic Evaluations

TruLens provides seven feedback templates specifically designed for evaluating agentic
systems. These templates assess different aspects of an agent's plan and execution,
giving you fine-grained insight into where your agent succeeds or struggles.

All agentic evaluators are available in `trulens.feedback.templates.agent` and operate
on the agent's trace — the full record of thinking, planning, and actions taken.

## Installation

```bash
pip install trulens trulens-providers-openai
```

## Quick start

```python
from trulens.core import TruSession
from trulens.apps.langchain import TruChain
from trulens.providers.openai import OpenAI
from trulens.feedback.templates.agent import (
    LogicalConsistency,
    ExecutionEfficiency,
    PlanAdherence,
    PlanQuality,
    ToolSelection,
    ToolCalling,
    ToolQuality,
)
from trulens.core.feedback import Feedback
from trulens.core.schema.select import Select

session = TruSession()
provider = OpenAI()

# Build feedback functions from agentic templates
f_logical_consistency = (
    Feedback(provider.run_template(LogicalConsistency), name="Logical Consistency")
    .on(Select.RecordCalls.agent.trace)
)
```

## Evaluators

### LogicalConsistency

**What it measures:** Whether every action, claim, and transition in the agent's trace is
explicitly justified by prior context. Flags hallucinations, unsupported assertions, and
logical leaps.

**Score range:** 0–3 (Likert scale, higher is better)

**Use when:** You need to verify that your agent's reasoning is grounded and coherent —
especially important for high-stakes domains like finance, legal, or medical.

**Score interpretation:**

| Score | Meaning |
|-------|---------|
| 3 | Every step is traceable to prior context; corrections are explicitly acknowledged |
| 1–2 | Occasional unsupported statements or implicit corrections |
| 0 | Frequent fabrications, contradictions, or unacknowledged errors |

```python
from trulens.feedback.templates.agent import LogicalConsistency

f_logical_consistency = Feedback(
    provider.run_template(LogicalConsistency),
    name="Logical Consistency",
)
```

---

### ExecutionEfficiency

**What it measures:** Whether the agent executed its plan without unnecessary repetition,
backtracking, or wasted computation. Penalizes redundant tool calls, preventable retries,
and poorly ordered steps.

**Score range:** 0–3

**Use when:** You want to reduce latency and cost by identifying agents that loop or
retry unnecessarily.

**Score interpretation:**

| Score | Meaning |
|-------|---------|
| 3 | All actions executed exactly once in optimal order |
| 1–2 | Some redundant steps or non-ideal ordering |
| 0 | Highly inefficient; dominated by loops or repeated failures |

```python
from trulens.feedback.templates.agent import ExecutionEfficiency

f_execution_efficiency = Feedback(
    provider.run_template(ExecutionEfficiency),
    name="Execution Efficiency",
)
```

---

### PlanAdherence

**What it measures:** Whether the agent's execution followed its plan step-by-step.
Penalizes skipped steps, unauthorized reordering, and undocumented deviations.

**Score range:** 0–3

**Use when:** You need the agent to follow a structured workflow exactly — e.g., a
multi-step research or data-processing pipeline.

**Score interpretation:**

| Score | Meaning |
|-------|---------|
| 3 | Every planned step executed and completed; deviations explicitly justified |
| 1–2 | Most steps followed; minor deviations with plausible explanations |
| 0 | Plan largely ignored; many steps skipped or replaced without justification |

```python
from trulens.feedback.templates.agent import PlanAdherence

f_plan_adherence = Feedback(
    provider.run_template(PlanAdherence),
    name="Plan Adherence",
)
```

---

### PlanQuality

**What it measures:** The intrinsic quality of the agent's plan — before execution.
Evaluates whether the plan correctly uses available tools, addresses the user's query,
and is structured logically. Does **not** evaluate whether the plan was followed.

**Score range:** 0–3

**Use when:** You want to evaluate the agent's planning capability separately from its
execution. Useful for debugging planning vs. execution failures.

**Score interpretation:**

| Score | Meaning |
|-------|---------|
| 3 | Well-structured plan; all steps necessary, justified, and feasible with available tools |
| 1–2 | Generally feasible; some steps lack justification or are partially implied |
| 0 | Plan fails to address the query or relies on non-existent tools |

```python
from trulens.feedback.templates.agent import PlanQuality

f_plan_quality = Feedback(
    provider.run_template(PlanQuality),
    name="Plan Quality",
)
```

---

### ToolSelection

**What it measures:** Whether the agent chose the *right* tools for each subtask given
the available tool descriptions. Focuses purely on selection suitability — not how the
tool was called or how efficiently it was used.

**Score range:** 0–3

**Use when:** You want to detect agents that consistently pick suboptimal or irrelevant
tools, or ignore mandated tools.

**Score interpretation:**

| Score | Meaning |
|-------|---------|
| 3 | Consistently selects the most suitable tool for each subtask |
| 1–2 | Generally appropriate; occasional missed opportunities or weak justification |
| 0 | Frequently selects ill-suited tools or ignores superior/mandated options |

```python
from trulens.feedback.templates.agent import ToolSelection

f_tool_selection = Feedback(
    provider.run_template(ToolSelection),
    name="Tool Selection",
)
```

---

### ToolCalling

**What it measures:** The quality of tool invocations — argument validity, semantic
appropriateness, precondition satisfaction, and how well the agent interprets tool
outputs. Does **not** judge which tool was chosen (see `ToolSelection`) or external
tool reliability (see `ToolQuality`).

**Score range:** 0–3

**Use when:** You want to find agents that pass invalid arguments, misread tool outputs,
or fail to handle tool errors gracefully.

**Score interpretation:**

| Score | Meaning |
|-------|---------|
| 3 | Syntactically valid, semantically appropriate calls; outputs correctly interpreted |
| 1–2 | Minor argument issues or shallow output use |
| 0 | Invalid/missing arguments, repeated schema violations, outputs ignored or fabricated |

```python
from trulens.feedback.templates.agent import ToolCalling

f_tool_calling = Feedback(
    provider.run_template(ToolCalling),
    name="Tool Calling",
)
```

---

### ToolQuality

**What it measures:** The reliability and output quality of the *tools themselves* as
observed in the trace — external errors (5xx, 429, 401), timeouts, flakiness, and
domain-specific output quality (e.g., search relevance). Independent of agent behavior.

**Score range:** 0–3

**Use when:** You want to separate tool-side failures from agent-side failures. Useful
for infrastructure monitoring and vendor evaluation.

**Score interpretation:**

| Score | Meaning |
|-------|---------|
| 3 | Tools respond reliably with relevant, complete outputs |
| 1–2 | Occasional external errors or weak outputs |
| 0 | Frequent errors, timeouts, rate limits, or persistently poor output quality |

```python
from trulens.feedback.templates.agent import ToolQuality

f_tool_quality = Feedback(
    provider.run_template(ToolQuality),
    name="Tool Quality",
)
```

---

## Using all seven evaluators together

```python
from trulens.core import TruSession
from trulens.providers.openai import OpenAI
from trulens.core.feedback import Feedback
from trulens.feedback.templates.agent import (
    LogicalConsistency,
    ExecutionEfficiency,
    PlanAdherence,
    PlanQuality,
    ToolSelection,
    ToolCalling,
    ToolQuality,
)

session = TruSession()
provider = OpenAI()

feedbacks = [
    Feedback(provider.run_template(LogicalConsistency), name="Logical Consistency"),
    Feedback(provider.run_template(ExecutionEfficiency), name="Execution Efficiency"),
    Feedback(provider.run_template(PlanAdherence), name="Plan Adherence"),
    Feedback(provider.run_template(PlanQuality), name="Plan Quality"),
    Feedback(provider.run_template(ToolSelection), name="Tool Selection"),
    Feedback(provider.run_template(ToolCalling), name="Tool Calling"),
    Feedback(provider.run_template(ToolQuality), name="Tool Quality"),
]
```

## Evaluator scope boundaries

The seven evaluators are designed to be **orthogonal** — each measures a distinct
aspect of agent behavior with minimal overlap:

| Evaluator | Scope |
|-----------|-------|
| LogicalConsistency | Reasoning coherence across the full trace |
| ExecutionEfficiency | Workflow sequencing and resource use |
| PlanAdherence | Execution fidelity to the stated plan |
| PlanQuality | Intrinsic plan quality (strategy, not outcome) |
| ToolSelection | Choice of tool per subtask |
| ToolCalling | Argument formation and output interpretation |
| ToolQuality | External tool/service reliability |

This separation lets you diagnose failures precisely: a low `PlanAdherence` score with
a high `PlanQuality` score means the agent planned well but failed to follow through.
A low `ToolCalling` score with a high `ToolSelection` score means the right tools were
chosen but invoked incorrectly.

## Related

- [Custom Feedback Functions](./feedback_implementations/custom_feedback_functions.ipynb)
- [Feedback Selectors](./feedback_selectors/index.md)
- [Agentic Evaluation Cookbook](../../../examples/cookbooks/agentic_evaluations.ipynb)
