# Agent GPA Framework

As demand grows for agentic systems that can plan, call tools and adapt, the stakes are rising. These systems now power enterprise-grade workflows and can serve thousands of business users, making one question especially critical: **Can you trust your agents to work as intended?**

An agent's answer may appear successful, but the path it took to get there may not be. Was the goal achieved efficiently? Did the plan make sense? Were the right tools used? Did the agent follow through? Without visibility into these steps, teams risk deploying agents that look reliable but create hidden costs in production. Inaccuracies can waste compute, inflate latency and lead to the wrong business decisions, all of which erode trust at scale.

Today's eval methods can fall short. They often judge only the final answer, missing the agent's decision-making process, which overlooks end-to-end performance. Ground-truth data sets with expected agent outcomes and trajectories annotated by experts are valuable but expensive to build and maintain. And outcome-focused benchmarks confirm whether an agent succeeded, but provide little insight into why it failed or how to fix it.

To address this gap, the Snowflake AI Research team developed the **Agent GPA (Goal-Plan-Action) framework**, available in the open source TruLens library, to evaluate agents across goals, plans and actions, surfacing internal errors such as hallucinations, poor tool use or missed plan steps.

## Understanding the Agent GPA Framework

Agent GPA evaluates agents across three critical phases of their reasoning and execution process — **Goal**, **Plan** and **Action** — using quantifiable metrics to capture what the agent produced and how it got there.

- **Goal**: Was the response relevant, grounded and accurate from the user's perspective?
- **Plan**: Did the agent design and follow a sound roadmap, selecting appropriate tools for each step?
- **Action**: Were those tools executed effectively and efficiently?

![Agent GPA Framework](../../assets/images/Agent_GPA.png)

Agent GPA uses LLM judges to score each metric, surfacing issues like hallucinations, reasoning gaps and inefficient tool use, making agent behavior transparent and easier to debug.

## Evaluation Metrics

### Alignment across Goal-Action

- **Answer correctness**: Does the agent's final answer align with the ground-truth reference? The LLM judge compares the agent's response to the verified answer to determine factual accuracy.
- **Answer relevance**: Is the agent's final answer relevant to the query? For example, a relevant result to "What is the weather today?" is "Cloudy." Whereas, "With a chance of meatballs" could be delicious, but sadly, irrelevant.
- **Groundedness**: Is the agent's final answer backed up by evidence from previously retrieved context?

### Alignment across Goal-Plan

- **Plan quality**: Did the agent design an effective roadmap to reach the goal? High-quality plans break down the problem into the right subtasks and assign the best tools for each.
- **Tool selection**: Did the agent choose the correct tool for each subtask? This is a special case of measuring plan quality.

### Alignment across Plan-Action

- **Plan adherence**: Did the agent follow through on its plan? Skipped, reordered or repeated steps often signal reasoning or execution errors.
- **Tool calling**: Are tool calls valid and complete, with correct parameters and appropriate use of outputs?

### Alignment across Goal-Plan-Action

- **Logical consistency**: Are the agent's steps coherent and grounded in prior context? This checks for contradictions, ignored instructions or reasoning errors.
- **Execution efficiency**: Did the agent reach the goal without wasted steps? This captures redundancies, superfluous tool calls or inefficient use of resources.

!!! note
    Tool-related evaluations in Agent GPA focus only on agent-controlled behavior, such as tool selection and tool calling. In production, teams often add enterprise-specific tool quality checks, such as retrieval relevance or API throughput, which fall outside the agent's control.

## Why Agent GPA Matters

Instead of treating agents as black boxes, Agent GPA makes their behavior observable and debuggable. These targeted insights help builders quickly refine agents for more accurate, reliable performance.

In benchmark testing on the [TRAIL](https://arxiv.org/abs/2505.08638) (Trace Reasoning and Agentic Issue Localization) benchmark built on the [GAIA](https://arxiv.org/abs/2311.12983) data set, Agent GPA judges consistently outperformed baseline LLM judges.

TRAIL is a publicly available dataset of 148 human-annotated agent traces with a formal taxonomy of error types spanning both single and multi-agent systems. It provides ground-truth error annotations across real-world tasks like software engineering and open-world information retrieval, making it an ideal testbed for validating evaluation frameworks like Agent GPA.

GAIA is a benchmark of 450+ non-trivial questions with unambiguous answers, designed to evaluate LLMs with augmented capabilities (tooling, search, multi-step reasoning). Questions are divided into three difficulty levels, where higher levels require greater autonomy and tool use. The TRAIL benchmark builds on GAIA by annotating the agent traces generated while attempting these questions, capturing where and how agents fail.

1. **95% error detection**, a 1.8x improvement over baseline methods.

![Agent GPA Error Coverage](../../assets/images/Agent_GPA_Error_Coverage.png)

2. **86% error localization**, compared to 49% for baseline judges, enabling faster debugging.

![Agent GPA Error Localization](../../assets/images/Agent_GPA_Error_Localization.png)


## Getting Started with Agent GPA

Here's a minimal example showing how to set up Agent GPA feedback functions:

```python
from trulens.core import Metric, Selector
from trulens.providers.openai import OpenAI

provider = OpenAI(model_engine="gpt-4.1")

f_plan_quality = Metric(
    implementation=provider.plan_quality_with_cot_reasons,
    selectors={
        "trace": Selector(trace_level=True),
    },
)

f_tool_selection = Metric(
    implementation=provider.tool_selection_with_cot_reasons,
    selectors={
        "trace": Selector(trace_level=True),
    },
)

f_execution_efficiency = Metric(
    implementation=provider.execution_efficiency_with_cot_reasons,
    selectors={
        "trace": Selector(trace_level=True),
    },
)
```

For step-by-step instructions on how to set up and use Agent GPA evaluations, check out these resources:

- Take the free DeepLearning.AI course: [Building and Evaluating Data Agents](https://learn.deeplearning.ai/courses/building-and-evaluating-data-agents/)
- Try out the [open source notebook with TruLens](../quickstarts/web-search-agent-evaluation.ipynb) to trace and evaluate agents
- Get started with [Cortex Agent Evaluations](https://www.snowflake.com/en/developers/guides/getting-started-with-cortex-agent-evaluations/) to evaluate agents in Snowflake

## Further Reading

- [Original Blog Post: What's Your Agent's GPA?](https://www.snowflake.com/en/engineering-blog/ai-agent-evaluation-gpa-framework/)
- [Agent GPA Research Paper](https://arxiv.org/abs/2510.08847)
