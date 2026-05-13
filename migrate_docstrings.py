#!/usr/bin/env python3
"""Migration script: Feedback( -> Metric( in docstrings only."""


def migrate_file(filepath, replacements):
    """Apply a list of (old, new) string replacements to a file."""
    with open(filepath, "r") as f:
        content = f.read()

    original = content
    for old, new in replacements:
        if old not in content:
            print(f"  WARNING: Pattern not found in {filepath}:")
            print(f"    {old[:80]}...")
            continue
        content = content.replace(old, new, 1)

    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        print(f"  MIGRATED: {filepath}")
    else:
        print(f"  NO CHANGES: {filepath}")


def migrate_llm_provider(base):
    fp = f"{base}/src/feedback/trulens/feedback/llm_provider.py"
    replacements = []

    # 1. context_relevance - .on_input().on(context).aggregate(np.mean)
    replacements.append((
        """            ```python
            from trulens.apps.langchain import TruChain
            context = TruChain.select_context(rag_app)
            feedback = (
                Feedback(provider.context_relevance,
                    criteria=criteria,
                    additional_instructions=additional_instructions,
                    examples=examples)
                .on_input()
                .on(context)
                .aggregate(np.mean)
                )
            ```""",
        """            ```python
            from trulens.core import Metric, Selector
            feedback = Metric(
                implementation=provider.context_relevance,
                name="Context Relevance",
                criteria=criteria,
                additional_instructions=additional_instructions,
                examples=examples,
                selectors={
                    "question": Selector.select_record_input(),
                    "context": Selector.select_context(
                        collect_list=False
                    ),
                },
                agg=np.mean,
            )
            ```""",
    ))

    # 2. context_relevance_with_cot_reasons
    replacements.append((
        """            ```python
            from trulens.apps.langchain import TruChain
            context = TruChain.select_context(rag_app)
            feedback = (
                Feedback(provider.context_relevance_with_cot_reasons,
                    criteria=criteria,
                    additional_instructions=additional_instructions,
                    examples=examples)
                .on_input()
                .on(context)
                .aggregate(np.mean)
                )
            ```""",
        """            ```python
            from trulens.core import Metric, Selector
            feedback = Metric(
                implementation=provider.context_relevance_with_cot_reasons,
                name="Context Relevance",
                criteria=criteria,
                additional_instructions=additional_instructions,
                examples=examples,
                selectors={
                    "question": Selector.select_record_input(),
                    "context": Selector.select_context(
                        collect_list=False
                    ),
                },
                agg=np.mean,
            )
            ```""",
    ))

    # 3. relevance - .on_input_output()
    replacements.append((
        """            ```python
            feedback = (
            Feedback(provider.relevance,
                criteria=criteria,
                additional_instructions=additional_instructions,
                examples=examples)
                .on_input_output()
                )
            ```

        Usage on RAG Contexts:
            ```python
            feedback = (
                Feedback(provider.relevance,
                    criteria=criteria,
                    additional_instructions=additional_instructions,
                    examples=examples)
                .on_input()
                .on(TruLlama.select_source_nodes().node.text) # See note below
                .aggregate(np.mean)
                )
            ```""",
        """            ```python
            from trulens.core import Metric, Selector
            feedback = Metric(
                implementation=provider.relevance,
                name="Answer Relevance",
                criteria=criteria,
                additional_instructions=additional_instructions,
                examples=examples,
                selectors={
                    "prompt": Selector.select_record_input(),
                    "response": Selector.select_record_output(),
                },
            )
            ```""",
    ))

    # 4. relevance_with_cot_reasons - .on_input().on_output()
    replacements.append((
        """            ```python
            feedback = (
                Feedback(provider.relevance_with_cot_reasons,
                    criteria=criteria,
                    additional_instructions=additional_instructions,
                    examples=examples)
                .on_input()
                .on_output()
                )
            ```""",
        """            ```python
            from trulens.core import Metric, Selector
            feedback = Metric(
                implementation=provider.relevance_with_cot_reasons,
                name="Answer Relevance",
                criteria=criteria,
                additional_instructions=additional_instructions,
                examples=examples,
                selectors={
                    "prompt": Selector.select_record_input(),
                    "response": Selector.select_record_output(),
                },
            )
            ```""",
    ))

    # 5. sentiment - .on_output()
    replacements.append((
        """            ```python
            feedback = (
                Feedback(provider.sentiment,
                    criteria=criteria,
                    additional_instructions=additional_instructions,
                    examples=examples)
                .on_output()
                )
            ```""",
        """            ```python
            from trulens.core import Metric, Selector
            feedback = Metric(
                implementation=provider.sentiment,
                name="Sentiment",
                criteria=criteria,
                additional_instructions=additional_instructions,
                examples=examples,
                selectors={
                    "text": Selector.select_record_output(),
                },
            )
            ```""",
    ))

    # 6. sentiment_with_cot_reasons - .on_output()
    replacements.append((
        """            ```python
            feedback = (
                Feedback(provider.sentiment_with_cot_reasons,
                    criteria=criteria,
                    additional_instructions=additional_instructions,
                    examples=examples)
                .on_output()
                )
            ```""",
        """            ```python
            from trulens.core import Metric, Selector
            feedback = Metric(
                implementation=provider.sentiment_with_cot_reasons,
                name="Sentiment",
                criteria=criteria,
                additional_instructions=additional_instructions,
                examples=examples,
                selectors={
                    "text": Selector.select_record_output(),
                },
            )
            ```""",
    ))

    # 7. model_agreement - .on_input_output()
    replacements.append((
        """            ```python
            feedback = Feedback(provider.model_agreement).on_input_output()
            ```""",
        """            ```python
            from trulens.core import Metric, Selector
            feedback = Metric(
                implementation=provider.model_agreement,
                name="Model Agreement",
                selectors={
                    "prompt": Selector.select_record_input(),
                    "response": Selector.select_record_output(),
                },
            )
            ```""",
    ))

    # 8-21. Langchain-eval style metrics (text param, .on_output())
    # These all follow the same pattern: Feedback(provider.X, criteria=..., additional_instructions=...).on_output()
    langchain_metrics = [
        "conciseness",
        "conciseness_with_cot_reasons",
        "correctness",
        "correctness_with_cot_reasons",
        "coherence",
        "coherence_with_cot_reasons",
        "harmfulness",
        "harmfulness_with_cot_reasons",
        "maliciousness",
        "maliciousness_with_cot_reasons",
        "helpfulness",
        "helpfulness_with_cot_reasons",
        "controversiality",
        "controversiality_with_cot_reasons",
        "misogyny",
        "misogyny_with_cot_reasons",
        "criminality",
        "criminality_with_cot_reasons",
        "insensitivity",
        "insensitivity_with_cot_reasons",
    ]

    for metric_name in langchain_metrics:
        display_name = metric_name.replace("_with_cot_reasons", "").title()
        old = f"""            ```python
            feedback = (
                Feedback(provider.{metric_name},
                    criteria=criteria,
                    additional_instructions=additional_instructions)
                .on_output()
                )
            ```"""
        new = f"""            ```python
            from trulens.core import Metric, Selector
            feedback = Metric(
                implementation=provider.{metric_name},
                name="{display_name}",
                criteria=criteria,
                additional_instructions=additional_instructions,
                selectors={{
                    "text": Selector.select_record_output(),
                }},
            )
            ```"""
        replacements.append((old, new))

    # 22. comprehensiveness_with_cot_reasons - .on_input_output()
    replacements.append((
        """            ```python
            feedback = Feedback(provider.comprehensiveness_with_cot_reasons).on_input_output()
            ```""",
        """            ```python
            from trulens.core import Metric, Selector
            feedback = Metric(
                implementation=provider.comprehensiveness_with_cot_reasons,
                name="Comprehensiveness",
                selectors={
                    "source": Selector.select_record_input(),
                    "summary": Selector.select_record_output(),
                },
            )
            ```""",
    ))

    # 23. stereotypes - .on_input_output()
    replacements.append((
        """            ```python
            feedback = Feedback(provider.stereotypes).on_input_output()
            ```""",
        """            ```python
            from trulens.core import Metric, Selector
            feedback = Metric(
                implementation=provider.stereotypes,
                name="Stereotypes",
                selectors={
                    "prompt": Selector.select_record_input(),
                    "response": Selector.select_record_output(),
                },
            )
            ```""",
    ))

    # 24. stereotypes_with_cot_reasons - .on_input_output()
    replacements.append((
        """            ```python
            feedback = Feedback(provider.stereotypes_with_cot_reasons).on_input_output()
            ```""",
        """            ```python
            from trulens.core import Metric, Selector
            feedback = Metric(
                implementation=provider.stereotypes_with_cot_reasons,
                name="Stereotypes",
                selectors={
                    "prompt": Selector.select_record_input(),
                    "response": Selector.select_record_output(),
                },
            )
            ```""",
    ))

    # 25. groundedness_measure_with_cot_reasons
    replacements.append((
        """            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI

            provider = OpenAI()

            f_groundedness = (
                Feedback(provider.groundedness_measure_with_cot_reasons)
                .on(context.collect())
                .on_output()
                )
            ```""",
        """            ```python
            from trulens.core import Metric, Selector
            from trulens.providers.openai import OpenAI

            provider = OpenAI()

            f_groundedness = Metric(
                implementation=provider.groundedness_measure_with_cot_reasons,
                name="Groundedness",
                selectors={
                    "source": Selector.select_context(
                        collect_list=True
                    ),
                    "statement": Selector.select_record_output(),
                },
            )
            ```""",
    ))

    # 26. groundedness_measure_with_cot_reasons_consider_answerability
    replacements.append((
        """            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI

            provider = OpenAI()

            f_groundedness = (
                Feedback(provider.groundedness_measure_with_cot_reasons_consider_answerability)
                .on(context.collect())
                .on_output()
                .on_input()
                )
            ```""",
        """            ```python
            from trulens.core import Metric, Selector
            from trulens.providers.openai import OpenAI

            provider = OpenAI()

            f_groundedness = Metric(
                implementation=provider.groundedness_measure_with_cot_reasons_consider_answerability,
                name="Groundedness",
                selectors={
                    "source": Selector.select_context(
                        collect_list=True
                    ),
                    "statement": Selector.select_record_output(),
                    "question": Selector.select_record_input(),
                },
            )
            ```""",
    ))

    # 27-33. Agentic metrics (trace param, Selector(trace_level=True))
    agentic_metrics = [
        "logical_consistency_with_cot_reasons",
        "execution_efficiency_with_cot_reasons",
        "plan_adherence_with_cot_reasons",
        "plan_quality_with_cot_reasons",
        "tool_selection_with_cot_reasons",
        "tool_calling_with_cot_reasons",
        "tool_quality_with_cot_reasons",
    ]

    for metric_name in agentic_metrics:
        display_name = (
            metric_name.replace("_with_cot_reasons", "")
            .replace("_", " ")
            .title()
        )
        old = f"""            ```python
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI

            provider = OpenAI()

            f_{metric_name.replace("_with_cot_reasons", "")} = (
                Feedback(provider.{metric_name})
                .on({{
                    "trace": Selector(trace_level=True),
                }})
            ```"""
        new = f"""            ```python
            from trulens.core import Metric, Selector
            from trulens.providers.openai import OpenAI

            provider = OpenAI()

            f_{metric_name.replace("_with_cot_reasons", "")} = Metric(
                implementation=provider.{metric_name},
                name="{display_name}",
                selectors={{
                    "trace": Selector(trace_level=True),
                }},
            )
            ```"""
        replacements.append((old, new))

    migrate_file(fp, replacements)


def migrate_openai_provider(base):
    fp = f"{base}/src/providers/openai/trulens/providers/openai/provider.py"
    with open(fp, "r") as f:
        content = f.read()

    # Read the file to understand exact patterns
    # These are moderation_* methods with Feedback(provider.X).on_output() or .on(...)
    replacements = []

    # Read file to find exact patterns
    lines = content.split("\n")
    # Find all Feedback( occurrences with context
    for i, line in enumerate(lines):
        if "Feedback(" in line:
            print(f"  OpenAI provider line {i + 1}: {line.strip()}")

    migrate_file(fp, replacements)


def migrate_openai_provider_manual(base):
    """Read and migrate openai provider manually."""
    fp = f"{base}/src/providers/openai/trulens/providers/openai/provider.py"
    with open(fp, "r") as f:
        content = f.read()

    print(f"\n  OpenAI provider Feedback( count: {content.count('Feedback(')}")


def main():
    base = "/Users/jreini/.snowflake/cortex/worktree/trulens/phase1-docstrings"

    print("=== Phase 1: llm_provider.py ===")
    migrate_llm_provider(base)

    print("\n=== Checking OpenAI provider ===")
    migrate_openai_provider_manual(base)


if __name__ == "__main__":
    main()
