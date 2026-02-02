---
categories:
  - General
date: 2026-02-05
---

# TruLens 2.6: Skills for AI Coding Assistants, PostgreSQL Support, and More

Building and evaluating LLM applications just got a whole lot easier. TruLens 2.6 brings powerful new capabilities that help you instrument, evaluate, and iterate on your AI apps faster than ever—whether you're working solo or with an AI coding assistant by your side.

<!-- more -->

---

## Agent Skills: Teach Your AI Assistant to Use TruLens

AI coding assistants like Cursor, Copilot, and Claude are transforming how we build software. But getting them to use specialized libraries correctly can be hit or miss. **TruLens 2.6 introduces Agent Skills**—structured knowledge files that teach AI assistants how to effectively instrument and evaluate your LLM applications.

### What Can You Do with Skills?

With the new skills system, your AI coding assistant can now:

**🔧 Instrument Any App Type**

- Set up TruLens tracing for LangChain, LangGraph, LlamaIndex, or custom Python apps
- Add custom spans to capture retrieval contexts, tool calls, and agent reasoning
- Use lambda-based attribute extraction for complex data structures
- Instrument third-party classes you can't modify with `instrument_method()`

**📊 Configure Evaluations Intelligently**

- Recommend the right metrics based on your app type (RAG Triad for retrieval apps, Agent GPA for agents)
- Set up feedback functions with proper selectors for your instrumented spans
- Handle `collect_list` correctly—individual evaluation vs. aggregated contexts
- Create custom metrics for domain-specific requirements

**🗂️ Curate Evaluation Datasets**

- Build ground truth datasets with expected responses and chunks
- Ingest external logs using VirtualRecord
- Persist and share evaluation data across your team

**▶️ Run and Analyze Evaluations**

- Execute evaluations and properly wait for async results
- Compare app versions on the leaderboard
- Run the TruLens dashboard

## AGENTS.md: TruLens is ready for AI contributors

We've added a comprehensive `AGENTS.md` file to the TruLens repository. This file provides AI coding assistants with essential context about the TruLens codebase, including:

- **Setup commands** for installation and development
- **Code style guidelines** including import conventions and docstring formats
- **Testing instructions** with markers and golden file regeneration
- **Project structure** explaining the modular package organization
- **Key patterns** for TruSession, app wrappers, and OTEL instrumentation
- **Troubleshooting tips** for common issues

### Why This Matters

When AI assistants understand your codebase structure, they:

- Generate code that follows your conventions
- Suggest appropriate import patterns
- Know where to find relevant files
- Understand how components interact

Whether you're contributing to TruLens or building on top of it, `AGENTS.md` helps your AI assistant be a more effective collaborator.

---

## PostgreSQL Support: Your Most Requested Feature

**PostgreSQL is the world's most popular open-source relational database**, and now TruLens fully supports it. This was one of our most requested features, and we're excited to deliver it.

### Why PostgreSQL?

- **Enterprise-ready**: PostgreSQL powers production workloads at companies of all sizes
- **Familiar infrastructure**: Many teams already run Postgres—now TruLens fits right in
- **Scalability**: Handle larger volumes of traces and evaluations
- **Rich ecosystem**: Leverage existing backup, monitoring, and management tools

### Simple Setup

```python
from trulens.core import TruSession

POSTGRES_URL = "postgresql://user:password@localhost:5432/trulens_db"

# That's it! TruLens now logs to your PostgreSQL database
session = TruSession(database_url=POSTGRES_URL)
```

TruLens automatically creates the required schema on first connection. Your traces, evaluations, and ground truth datasets are all stored in Postgres and queryable with standard SQL tools and the TruLens dashboard will read from Postgres seamlessly.

---

## Reliable Feedback Result Retrieval

As TruLens adoption grows, we're seeing more teams run evaluations in automated scripts and CI/CD pipelines—not just interactive notebooks. This shift means you need a reliable way to wait for evaluation results before making pass/fail decisions or moving to the next pipeline stage.

The new `retrieve_feedback_results()` method properly waits for all feedback evaluations to complete:

```python
with tru_rag as recording:
    for q in queries:
        rag.query(q)

# This will reliably wait for ALL feedback results
feedback_results = recording.retrieve_feedback_results(timeout=300)
print(feedback_results)
```

* This method handles:
- Records being written to the database
- Feedback evaluations completing
- Results becoming available

---

## Get Started

Ready to try TruLens 2.6?

```bash
pip install trulens --upgrade
```

### Quick Links

- [TruLens Documentation](https://www.trulens.org/)
- [GitHub Repository](https://github.com/truera/trulens)
- [Quickstart Notebook](https://github.com/truera/trulens/tree/main/examples/quickstart)
- [PostgreSQL Setup Guide](https://www.trulens.org/component_guides/logging/where_to_log/postgres/)

---
**Have feedback or feature requests?** Open an [issue](https://github.com/truera/trulens/issues) or [discussion](https://github.com/truera/trulens/discussions) on GitHub.
---
