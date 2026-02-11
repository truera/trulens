---
categories:
  - General
date: 2026-02-03
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

**Explore the skills:** [TruLens Skills on GitHub](https://github.com/truera/trulens/tree/main/skills)

**Using with Claude Code:** Copy the `skills/` directory into your project and run `/add-skill skills/SKILL.md` to enable TruLens evaluation workflows.

## AGENTS.md: Contribute to TruLens with AI Assistance

While Skills help you *use* TruLens, `AGENTS.md` helps you *contribute* to TruLens. We've added this file alongside an updated contribution guide to make it easier than ever to contribute to the project with AI coding assistants.

When you use your favorite coding assistant to work on TruLens, your assistant automatically understands:

- **Code style conventions** — 80-character lines, Google-style docstrings, and our module import patterns (e.g., `from trulens.schema import record as record_schema`)
- **How to run tests** — Unit test commands, test markers like `@pytest.mark.optional`, and how to regenerate golden files
- **Project structure** — Where to find core abstractions, providers, app integrations, and connectors
- **Development workflow** — `poetry install`, `make format`, `make lint`, and pre-commit hooks

This means your AI assistant can help you write code that passes CI on the first try, follows our conventions, and fits naturally into the codebase architecture.

**Want to contribute?** Check out our updated [contribution guide](https://www.trulens.org/contributing/) and let your AI assistant handle the style details.

---

## PostgreSQL Support

**PostgreSQL is the world's most popular open-source relational database**—trusted by millions of developers and powering everything from startups to Fortune 500 companies. Now TruLens fully supports it.

### Why This Matters

- **You're probably already using it**: PostgreSQL dominates the database landscape, consistently ranking #1 in developer surveys. Chances are your team already has Postgres infrastructure in place.
- **Enterprise-grade reliability**: Battle-tested at scale with ACID compliance, robust replication, and decades of production hardening
- **Rich ecosystem**: Leverage your existing backup, monitoring, and management tools—no new operational overhead
- **SQL queryability**: Analyze your traces and evaluations with standard SQL alongside your other application data

!!! example "Connect to PostgreSQL"

    ```python
    from trulens.core import TruSession

    POSTGRES_URL = "postgresql://user:password@localhost:5432/trulens_db"

    # That's it! TruLens now logs to your PostgreSQL database
    session = TruSession(database_url=POSTGRES_URL)
    ```

TruLens automatically creates the required schema on first connection. Your traces, evaluations, and ground truth datasets are all stored in Postgres and queryable with standard SQL tools and the TruLens dashboard will read from Postgres seamlessly.

**Learn more:** [PostgreSQL Documentation](https://www.trulens.org/component_guides/logging/where_to_log/log_in_postgres/) | [Example Notebook](https://github.com/truera/trulens/tree/main/examples/expositional/logging/log_in_postgres.ipynb)

---

## Reliable Feedback Result Retrieval: Your Most Requested Feature

As TruLens adoption grows, more teams are running evaluations in automated scripts and CI/CD pipelines—not just interactive notebooks. This was our most requested feature: a reliable way to wait for evaluation results before making pass/fail decisions or moving to the next pipeline stage.

We've added two methods to support this workflow:

- **`recording.retrieve_feedback_results()`** — Wait for evaluations to complete and return the results as a DataFrame
- **`session.wait_for_feedback_results()`** — Wait for specific feedback evaluations by record ID and feedback name

!!! example "Retrieve Feedback Results"

    ```python
    with tru_rag as recording:
        for q in queries:
            rag.query(q)

    # Wait and retrieve results as a DataFrame
    feedback_results = recording.retrieve_feedback_results(timeout=300)
    print(feedback_results)
    ```

!!! example "Wait for Specific Feedbacks"

    ```python
    # Wait for specific feedbacks on specific records
    session.wait_for_feedback_results(
        record_ids=record_ids,
        feedback_names=["Answer Relevance", "Groundedness"],
        timeout=300
    )
    ```

These methods handle:

- Records being written to the database
- Feedback evaluations completing
- Results becoming available

---

## Get Started

Ready to try TruLens 2.6?

!!! example "Install TruLens"

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
