---
skill_spec_version: 0.1.0
name: trulens-evaluation-workflow
version: 1.0.0
description: Systematically evaluate your LLM application with TruLens
tags: [trulens, llm, evaluation, workflow, orchestration]
---

# TruLens Evaluation Workflow

A systematic approach to evaluating your LLM application.

## When to Use This Skill

Use this skill when you want to:
- Set up comprehensive evaluation for a new LLM app
- Improve an existing app's evaluation coverage
- Understand the full TruLens workflow
- Know which sub-skill to use for your current task

## The Evaluation Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TruLens Evaluation Workflow                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. INSTRUMENT          2. CURATE            3. CONFIGURE       │
│   ┌──────────────┐      ┌──────────────┐     ┌──────────────┐   │
│   │ Capture data │  →   │ Build test   │  →  │ Choose       │   │
│   │ from your    │      │ datasets     │     │ metrics      │   │
│   │ app          │      │              │     │              │   │
│   └──────────────┘      └──────────────┘     └──────────────┘   │
│         ↓                                           ↓            │
│         └─────────────────────┬─────────────────────┘            │
│                               ↓                                  │
│                      4. RUN & ANALYZE                            │
│                      ┌──────────────┐                            │
│                      │ Execute evals│                            │
│                      │ & iterate    │                            │
│                      └──────────────┘                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Sub-Skills Reference

| Step | Skill | When to Use |
|------|-------|-------------|
| **1. Instrument** | `instrumentation/` | Setting up a new app, adding custom spans, capturing specific data for evals |
| **2. Curate** | `dataset-curation/` | Creating test datasets, storing ground truth, ingesting external logs |
| **3. Configure** | `evaluation-setup/` | Choosing metrics (RAG triad vs Agent GPA), setting up feedback functions |
| **4. Run** | `running-evaluations/` | Executing evaluations, viewing results, comparing versions |

## Interactive Workflow Guide

**Answer these questions to find where to start:**

### Where are you in the process?

**"I have a new LLM app that isn't instrumented yet"**
→ Start with `instrumentation/` skill

**"My app is instrumented but I don't have test data"**
→ Go to `dataset-curation/` skill

**"I have data but haven't set up evaluations"**
→ Go to `evaluation-setup/` skill

**"Everything is set up, I just need to run evals"**
→ Go to `running-evaluations/` skill

---

### What's your immediate goal?

**"I want to see traces of my app's execution"**
→ Use `instrumentation/` - capture spans and view in dashboard

**"I want to evaluate my RAG's retrieval quality"**
→ Use `evaluation-setup/` - configure RAG Triad metrics

**"I want to evaluate my agent's tool usage"**
→ Use `evaluation-setup/` - configure Agent GPA metrics

**"I want to compare two versions of my app"**
→ Use `running-evaluations/` - version comparison pattern

**"I want to evaluate against known correct answers"**
→ Use `dataset-curation/` - create ground truth dataset

---

## Quick Start Paths

### Path A: Evaluate a RAG App

1. **Instrument** → Wrap with `TruLlama` or `TruChain`
2. **Configure** → Set up RAG Triad (context relevance, groundedness, answer relevance)
3. **Run** → Execute queries and view leaderboard

### Path B: Evaluate an Agent

1. **Instrument** → Wrap with `TruGraph`
2. **Configure** → Set up Agent GPA metrics
3. **Run** → Execute tasks and analyze traces

### Path C: Regression Testing

1. **Curate** → Create ground truth test dataset
2. **Configure** → Add ground truth agreement metric
3. **Run** → Compare versions against test set

### Path D: Production Monitoring

1. **Instrument** → Add custom attributes for key data
2. **Configure** → Set up metrics for production concerns
3. **Run** → Continuously evaluate production traffic

---

## Common Questions

**"Do I need to use all four skills?"**
No. Instrumentation and evaluation-setup are essential. Dataset-curation is optional (for ground truth comparisons). Running-evaluations is needed to execute and view results.

**"What order should I use them?"**
Generally: Instrument → (optionally) Curate → Configure → Run. But you can revisit any step as needed.

**"Can I add more evaluations later?"**
Yes. You can always add new feedback functions and re-run evaluations on existing traces.

**"How do I know if my app is a RAG or Agent?"**
- **RAG**: Retrieves documents/context, generates grounded responses
- **Agent**: Uses tools, makes decisions, may involve planning

If your app does both (e.g., agentic RAG), use metrics from both categories.

---

## Getting Help

If you're unsure which skill to use, describe your goal and I'll guide you to the right one.
