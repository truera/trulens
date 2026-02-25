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

In benchmark testing on the TRAIL/GAIA data set, Agent GPA judges consistently outperformed baseline LLM judges:

- **Consistent, trustworthy coverage** with Agent GPA judges matching 570 human annotated failures across all error severities.
- **Increased confidence in LLM evals** with 95% error detection, a 1.8x improvement over baseline methods.
- **Faster debugging** with 86% error localization, compared to 49% for baseline judges.

### Error Impact Levels

The TRAIL benchmark categorizes errors into three impact levels:

- **Low-impact errors**: Minor issues, such as typos, formatting mistakes, or redundant steps, that don't affect the correctness of the final answer but reveal small inefficiencies or surface-level reasoning flaws.
- **Medium-impact errors**: Partial reasoning or execution mistakes, such as choosing an inappropriate tool or skipping a step that slightly alters the outcome.
- **High-impact errors**: Severe failures that lead to incorrect or fabricated results, including hallucinated data, broken reasoning chains or major plan deviations.

## Getting Started with Agent GPA

For step-by-step instructions on how to set up and use Agent GPA evaluations, check out these resources:

- Take the free DeepLearning course: [Building and Evaluating Data Agents](https://www.deeplearning.ai/)
- Try out the [open source notebook with TruLens](../quickstarts/web-search-agent-evaluation.ipynb) to trace and evaluate agents
- Use [Snowflake AI Observability](https://docs.snowflake.com/) to trace and evaluate agents in Snowflake

## Further Reading

- [Original Blog Post: What's Your Agent's GPA?](https://www.snowflake.com/en/engineering-blog/ai-agent-evaluation-gpa-framework/)
- [Agent GPA Research Paper](https://arxiv.org/abs/2505.17567)
