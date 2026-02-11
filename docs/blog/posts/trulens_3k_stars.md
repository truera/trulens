---
categories:
  - General
date: 2026-01-08
---

# Celebrating 3,000 Stars: Evolving TruLens for Real-World Agent Workflows

When we started building TruLens, our mission was clear: make it easier to evaluate, trace, and improve LLM-based applications so they can be trusted in production.
Reaching 3,000 GitHub stars is a meaningful milestone, and we are genuinely grateful to everyone in the community who has contributed ideas, submitted issues and shared their feedback. Thank you. We would not be here without this community behind us.

It also reflects why TruLens has become even more relevant today in the age of AI agents. AI agents are more complex than the previous generation of AI applications, and small errors quickly compound into larger failures. As more teams adopt agents in real workflows, the demand for reliable tracing and evaluation has never been higher.

For AI agents, traces are dramatically more useful as they can uncover when agent execution diverges from expectation. As these traces grow in size and complexity, automated evaluation becomes just as critical. Instead of manually digging through the trace to find mistakes, LLM judges can quickly find errors so you can get back to improving your agent.
In short, tracing and evaluation now play a central role in getting reliable agents into production.

This post highlights how we’re evolving TruLens for these real-world agent workflows, including new evaluation methods, support for reasoning models, and improved handling of MCP-based tool calls.

## Measuring Goal-Plan-Action Alignment

One challenge we’ve observed with teams building agents is that annotating agent traces is extremely time intensive and difficult. But without annotation, getting signals on ways to improve the agent is practically impossible.

To address this challenge, we’ve introduced a new framework for evaluating the alignment of an agent’s goal, plan and actions which we’ve dubbed the [Agent’s GPA.](https://www.snowflake.com/en/engineering-blog/ai-agent-evaluation-gpa-framework/)

We [benchmarked this framework to cover 95% of internal agent errors on the open-source TRAIL dataset](https://arxiv.org/abs/2510.08847), composed of agent traces for software engineering and data tasks. This framework provides an exciting new set of reference-free metrics that review agent traces and identify ways to improve the agent.

![TruLens Agent GPA](../assets/agent_gpa.png)

## Enabling reasoning models for agent evals

A major paradigm shift in the industry has been the rise of reasoning models. These powerful models often have different API shapes and output formats. We enabled support for [Deepseek](https://github.com/truera/trulens/pull/2191) models, and OpenAI's [GPT-5](https://github.com/truera/trulens/pull/2189) and [o-series](https://github.com/truera/trulens/pull/2138), so they can be used as LLM judges in TruLens and enable richer reasoning for evaluation.

Because agent traces are massively more complex than prior evolutions of LLM applications, reasoning models are particularly useful for evaluation. In internal benchmarks of agent evaluation metrics, particularly logical consistency, we saw significant improvements for using reasoning models (such as GPT-4o) compared to oneshot (such as GPT-4.1).

## MCP support

Many of today’s agent systems now use the Model Context Protocol to connect tools to agents.

To ensure TruLens fits naturally into these emerging workflows, we added a new [MCP span type](https://www.trulens.org/otel/semantic_conventions/) so that tool calls can be properly annotated as MCP-based tools. This new span type allows for finer segmentation of failure modes and faster agent debugging.

## The road ahead

As we look forward, we will double down on improving your ability to debug and improve AI agents with tools for tracing, evaluation, and optimization. If you have ideas or feature requests - [please create a GitHub discussion thread](https://github.com/truera/trulens/discussions/new/choose).

If you haven’t joined the community yet:

⭐ [Star us on GitHub](https://github.com/truera/trulens)
🧠 Try our free course on DeepLearning.ai: [Building and Evaluating Data Agents](https://www.deeplearning.ai/short-courses/building-and-evaluating-data-agents/)
 📚 Get started and check out [TruLens docs](https://www.trulens.org/)

Here’s to the next 3,000 stars.
