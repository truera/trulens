# LLM Judge Alignment

An LLM judge is only useful when its scores reflect the decisions people would
make for the same task. Model quality alone does not guarantee that alignment.
The rubric, additional instructions, examples, score scale, and decision
threshold can all change the result.

LLM judges can approach human agreement on some pairwise preference tasks, but
they also exhibit position, verbosity, self-enhancement, and reasoning biases
([Zheng et al., 2023](https://arxiv.org/abs/2306.05685)). Across a broader set of
20 NLP evaluation tasks, alignment varies substantially by model, dataset,
evaluated property, and annotator expertise
([Bavaresco et al., 2024](https://arxiv.org/abs/2406.18403)). Treat alignment as
something to measure for a specific judge configuration and target, not as an
inherent property of the underlying model.

Use a labeled golden set to develop the judge as you would any other measured
system: establish a baseline, inspect failures, change one dimension, validate
the change, and reserve held-out data for the final decision.

## The judge development loop

1. **[Define the target and choose what to change](#define-the-target-and-choose-what-to-change).**
   Write down what the score means and what it excludes.
   If several people label the data, use one rubric and adjudicate disagreements.
2. **[Build a representative golden set](#build-a-representative-golden-set).**
   Include clear successes, clear failures, borderline cases, and important
   production slices. Difficult cases matter because judges that perform well on
   ordinary preference data can fail on pairs requiring factual, logical,
   mathematical, or coding correctness
   ([Tan et al., 2024](https://arxiv.org/abs/2410.12784)).
3. **[Freeze development, validation, and held-out splits](#build-a-representative-golden-set).**
   Tune on development, select among candidates on validation, and use held-out
   labels once for the final comparison.
4. **[Run a baseline](#run-a-baseline).** First
   [represent the complete judge configuration](#represent-the-complete-judge-configuration),
   then keep the provider, model, temperature, parser, and examples fixed while
   measuring the current judge.
5. **[Diagnose before changing anything](#diagnose-before-changing-anything).**
   Check absolute error, ranking, calibration, thresholds, score distribution,
   and the largest misses.
6. **[Change one dimension](#change-one-dimension-optimize-or-ensemble).** Revise
   the rubric, add instructions, select examples, replace the model, or use an
   ensemble. Do not change several at once unless you are prepared to attribute
   the result to the complete configuration.
7. **[Compare paired results](#compare-paired-results).** Run both variants over
   the same rows. Recheck every production threshold even when continuous
   alignment improves.
8. **[Confirm on held-out data](#confirm-on-held-out-data).** Record the complete
   judge configuration and keep the per-row outputs so the result can be
   reproduced.

The worked
[judge alignment experiment](../../blog/posts/improving_llm_judge_alignment.md)
applies this protocol to SummEval and shows why better calibration can still
worsen agreement at one decision threshold.

## Define the target and choose what to change

| Need | Change | Validate with |
|---|---|---|
| The judge applies the wrong definition | [`criteria`][trulens.feedback.llm_provider.LLMProvider.relevance] | [`CriteriaABTest`][trulens.benchmark.criteria_ab_test.CriteriaABTest] and [`AlignmentReport`][trulens.benchmark.alignment_report.AlignmentReport] |
| The definition is right but domain rules are missing | [`additional_instructions`][trulens.feedback.llm_provider.LLMProvider.relevance] | [`CriteriaABTest`][trulens.benchmark.criteria_ab_test.CriteriaABTest] and slice-level misses |
| `criteria` is hard to express without demonstrations | [`examples`][trulens.feedback.llm_provider.LLMProvider.relevance] | [`FewShotOptimizer`][trulens.feedback.optimize.FewShotOptimizer], then held-out [`AlignmentReport`][trulens.benchmark.alignment_report.AlignmentReport] |
| Scores collapse into a narrow range | [`criteria`][trulens.feedback.llm_provider.LLMProvider.relevance], [`examples`][trulens.feedback.llm_provider.LLMProvider.relevance], or [`output_space`][trulens.feedback.llm_provider.LLMProvider.relevance] | [`ScoreDistributionAnalyzer`][trulens.benchmark.score_distribution.ScoreDistributionAnalyzer] |
| A cheaper or newer model might replace the current judge | [`LLMProvider`][trulens.feedback.llm_provider.LLMProvider] or [`model_engine`][trulens.feedback.llm_provider.LLMProvider] | [`CrossModelAlignment`][trulens.benchmark.cross_model_alignment.CrossModelAlignment] plus ground truth |
| One judge is too noisy or biased | [`Jury`][trulens.feedback.jury.Jury] with diverse models | [`AlignmentReport`][trulens.benchmark.alignment_report.AlignmentReport] for the aggregate and jurors |
| Prompt changes are exhausted and enough labels exist | Fine-tune the [`model_engine`][trulens.feedback.llm_provider.LLMProvider] used by [`LLMProvider`][trulens.feedback.llm_provider.LLMProvider] | Fresh held-out labels and [`AlignmentReport`][trulens.benchmark.alignment_report.AlignmentReport] threshold analysis |

`criteria` replaces the metric's default definition.
`additional_instructions` preserves that definition and adds domain-specific
rules. Prefer additional instructions when the stock metric still describes the
target; prefer custom criteria when the target itself is different.

This configuration matters empirically. G-Eval found that explicit evaluation
criteria and evaluation steps improved correlation with human judgments on
summarization and dialogue tasks
([Liu et al., 2023](https://arxiv.org/abs/2303.16634)). AutoCalibrate similarly
uses human-labeled examples to construct and select task-specific criteria
([Liu et al., 2023](https://arxiv.org/abs/2309.13308)). These results motivate
testing `criteria` and `additional_instructions`; they do not remove the need to
validate the resulting scores and thresholds on your own labels.

!!! warning "Do not define success with the judge under test"

    Human labels define the target. Do not use an LLM to create the labels used
    to prove that the same judge aligns.

## Build a representative golden set

The benchmark utilities accept rows with the judge inputs and a human label
normalized to `[0, 1]`:

```python
golden_set = [
    {
        "example_id": "policy-001",
        "query": "What is the return policy?",
        "expected_response": "Returns are accepted within 30 days.",
        "expected_score": 1.0,
        "split": "development",
        "metadata": {"slice": "policy"},
    },
    {
        "example_id": "policy-002",
        "query": "What is the return policy?",
        "expected_response": "Contact sales for enterprise pricing.",
        "expected_score": 0.0,
        "split": "development",
        "metadata": {"slice": "policy"},
    },
    {
        "example_id": "shipping-001",
        "query": "How long does standard shipping take?",
        "expected_response": "Standard shipping takes three to five business days.",
        "expected_score": 1.0,
        "split": "validation",
        "metadata": {"slice": "shipping"},
    },
    {
        "example_id": "shipping-002",
        "query": "How long does standard shipping take?",
        "expected_response": "Returns are accepted within 30 days.",
        "expected_score": 0.0,
        "split": "held_out",
        "metadata": {"slice": "shipping"},
    },
]

development_set = [row for row in golden_set if row["split"] == "development"]
validation_set = [row for row in golden_set if row["split"] == "validation"]
held_out_set = [row for row in golden_set if row["split"] == "held_out"]
```

The rows are intentionally small for illustration; use enough independently
labeled examples to represent the production slices and score ranges you intend
to evaluate.

If production records already exist in TruLens, use
[`GoldenSetGenerator`][trulens.benchmark.golden_set_generator.GoldenSetGenerator]
to sample annotation candidates:

```python
from trulens.benchmark.golden_set_generator import GoldenSetGenerator
from trulens.core import TruSession

generator = GoldenSetGenerator(TruSession(), seed=42)
sample = generator.sample(
    n=50,
    app_name="my_rag_app",
    strategy="stratified",
    feedback_name="relevance",
)

# Freeze a reproducible 60/20/20 split before tuning any judge.
sample_df = (
    sample.to_df().sample(frac=1, random_state=42).reset_index(drop=True)
)
sample_df["split"] = (
    ["development"] * 30 + ["validation"] * 10 + ["held_out"] * 10
)

# After human annotators populate expected_score directly in sample_df:
annotated = generator.load_annotations(sample_df)

for split_name in ("development", "validation", "held_out"):
    generator.save_golden_set(
        f"judge_golden_v1_{split_name}",
        annotated[annotated["split"] == split_name],
        dataset_metadata={
            "version": "v1",
            "split": split_name,
            "criteria": "relevance-v2",
        },
    )
```

Choose `random` for an unbiased traffic sample, `stratified` to cover low,
medium, and high judge scores, or `uncertainty` to focus near a decision
boundary. `GoldenSetGenerator` does not assign or protect evaluation splits. The
example therefore freezes the split with a seeded DataFrame shuffle, retains the
assignment while human labels are added to the DataFrame, and persists each split
as a separately named TruLens dataset. Do not recompute the split when rerunning
the experiment.

!!! warning "Keep held-out labels held out"

    Tune on development data and select candidates on validation data. Do not
    inspect or tune against the held-out split before the final comparison.

## Represent the complete judge configuration

TruLens does not require a separate judge-specification class. Keep each
configuration as arguments that can construct a `Metric`, alongside the metadata
needed to reproduce the run:

```python
from trulens.providers.openai import OpenAI

provider = OpenAI(model_engine="gpt-4o-mini")
strict_criteria = (
    "A response is relevant only when it directly answers the prompt."
)
domain_instructions = "Do not reward fluency when the response is incomplete."
selected_examples = [
    ({"prompt": "What is 2 + 2?", "response": "4"}, 3),
    ({"prompt": "What is 2 + 2?", "response": "Paris"}, 0),
]

judge_configs = {
    "baseline": {
        "metric": {
            "implementation": provider.relevance_with_cot_reasons,
            "name": "relevance-default",
            "temperature": 0.0,
        },
        "metadata": {
            "provider": "openai",
            "model_engine": "gpt-4o-mini",
        },
    },
    "strict": {
        "metric": {
            "implementation": provider.relevance_with_cot_reasons,
            "name": "relevance-strict",
            "criteria": strict_criteria,
            "additional_instructions": domain_instructions,
            "examples": selected_examples,
            "min_score_val": 0,
            "max_score_val": 3,
            "temperature": 0.0,
        },
        "metadata": {
            "provider": "openai",
            "model_engine": "gpt-4o-mini",
        },
    },
}
```

Also record the dataset version, split assignment, request date, retries, and
parser behavior. A model name and an aggregate score are not enough to reproduce
an LLM judge experiment.

## Run a baseline

Construct a [`Metric`][trulens.core.Metric] directly from each configuration:

```python
from trulens.core import Metric

judges = {
    name: Metric(**config["metric"]) for name, config in judge_configs.items()
}

score, metadata = judges["strict"](
    prompt="What is the return policy?",
    response="Returns are accepted within 30 days.",
)
```

`Metric.__call__` forwards the configured `criteria`,
`additional_instructions`, `examples`, score range, and `temperature` to the
implementation. Methods ending in `_with_cot_reasons` return `(score, metadata)`;
plain methods such as `provider.relevance` return only the normalized score.

The direct call is useful for a smoke test, not as the default batch execution
strategy. Choose the native TruLens API based on whether you need to execute a
judge, compare configurations, persist results, or diagnose existing scores:

| API | Use when | Input | Executes judges? | Persists results? | Primary output |
|---|---|---|---|---|---|
| [`Metric`][trulens.core.Metric] direct call | Smoke-testing one configuration on one example | Arguments matching the implementation signature | Yes, one call | No | Score, plus optional reason metadata |
| [`CriteriaABTest`][trulens.benchmark.criteria_ab_test.CriteriaABTest] | Comparing exactly two callable configurations on one small labeled set | `golden_set`, two variants | Yes, synchronously per row | No | Paired scores, ground-truth metrics, disagreements, score-shift test, winner |
| [`CrossModelAlignment`][trulens.benchmark.cross_model_alignment.CrossModelAlignment] | Comparing the same feedback method across several providers or `model_engine` values | `golden_set`, judge providers | Yes, synchronously per row | No | Pairwise agreement, score bias, ground-truth metrics, recommendations |
| [`Run`][trulens.core.run.Run] with `LOG_INGESTION` | Evaluating existing questions and responses with multiple `Metric` objects | DataFrame or table mapped to span attributes | Yes, during `compute_metrics()` | Yes | Traces, metric scores, record details, dashboard data |
| [`Run`][trulens.core.run.Run] with `APP_INVOCATION` | Invoking an app over a dataset, then evaluating its new outputs | DataFrame or table plus app recorder | Yes, app and metrics | Yes | App traces, outputs, metric scores, dashboard data |
| [`AlignmentReport`][trulens.benchmark.alignment_report.AlignmentReport] | Diagnosing judge scores that have already been computed | Aligned predicted scores and human labels | No | No | Error, rank agreement, calibration, thresholds, distributions, worst misses |
| [`ScoreDistributionAnalyzer`][trulens.benchmark.score_distribution.ScoreDistributionAnalyzer] | Running one judge while checking score spread and calibration | One feedback callable and `golden_set` | Yes, synchronously per row | No | Distribution statistics, calibration curve, pathology flags |
| [`FewShotOptimizer`][trulens.feedback.optimize.FewShotOptimizer] | Selecting demonstrations from a candidate pool using a separate labeled evaluation set | Candidates, evaluation rows, feedback callable | Yes, many candidate trials | No | Selected examples and optimization score |

`CriteriaABTest`, `CrossModelAlignment`, `ScoreDistributionAnalyzer`, and
`FewShotOptimizer` are focused development utilities. They perform direct
provider calls and keep results in memory. Use a `Run` when evaluation results
must be persisted, inspected through records, or displayed in the dashboard.
Use `AlignmentReport` after execution, regardless of which execution path
produced the scores.

For the two configurations above, use `CriteriaABTest` as the batch runner. The
`Metric` objects retain the settings constructed from `judge_configs`:

```python
from trulens.benchmark.criteria_ab_test import CriteriaABTest

comparison = CriteriaABTest(
    golden_set=development_set,
    variant_a={"fn": judges["baseline"], "name": "baseline"},
    variant_b={"fn": judges["strict"], "name": "strict"},
)
comparison_report = comparison.run()

judge_outputs = {
    comparison_report.name_a: comparison_report.scores_a,
    comparison_report.name_b: comparison_report.scores_b,
}
```

These aligned score arrays can be passed directly to `AlignmentReport`.
`CriteriaABTest` and `CrossModelAlignment` provide comparison semantics and row
alignment, but they still issue provider calls for each row and configuration.
Use them for focused judge-development sets, not as a high-throughput execution
engine.

For production-scale or persisted evaluation, prefer the Run API. A run invokes
the app over a table or DataFrame, then computes all configured `Metric` objects
over the ingested traces:

```python
import pandas as pd
from trulens.core.run import RunConfig

golden_set_df = pd.DataFrame(development_set).rename(
    columns={"example_id": "input_id"}
)

run = tru_app.add_run(
    run_config=RunConfig(
        run_name="judge-alignment-v2",
        dataset_name="judge_golden_set",
        source_type="DATAFRAME",
        mode="LOG_INGESTION",
        dataset_spec={
            "input_id": "input_id",
            "record_root.input": "query",
            "record_root.output": "expected_response",
        },
    )
)
run.start(input_df=golden_set_df)

# After run.get_status() reports INVOCATION_COMPLETED:
run.compute_metrics(list(judges.values()))
records = run.get_records()
```

Runs store traces and scores, support multiple metrics in one batch workflow, and
make results available to the dashboard. When the app outputs already exist,
use the data-replay pattern instead of invoking the app again. Both workflows
preserve normal `Metric` selectors and result logging; use direct provider calls
only for quick checks or small, purpose-built comparison utilities.

Keep outputs aligned with stable example IDs. Cache real provider outputs when
runs are expensive, and record errors rather than silently substituting simulated
scores. Use separate development, validation, and held-out datasets so executing
a judge does not accidentally expose held-out labels during tuning.

!!! warning "Report measured outputs only"

    Do not replace failed or unconfigured provider calls with simulated scores.
    Record the failure or skip the configuration explicitly; otherwise benchmark
    metrics no longer describe the judge that was run.

## Diagnose before changing anything

When scores have already been computed, `AlignmentReport` is the broadest
diagnostic:

```python
from trulens.benchmark import AlignmentReport

report = AlignmentReport(
    predicted_scores=judge_outputs["strict"],
    true_labels=[row["expected_score"] for row in development_set],
    examples=development_set,
    threshold=0.5,
    thresholds=[0.3, 0.5, 0.7],
    n_bins=5,
    top_n=10,
)

report.print_summary()
frames = report.to_dataframe()
figure = report.plot()
```

Read the outputs together:

- **MAE and Brier score** measure score error and calibration.
- **Spearman and Kendall** measure whether the judge ranks examples correctly.
- **Kappa and confusion matrices** measure decisions at configured thresholds.
- **Calibration and score distributions** reveal score inflation or collapse.
- **Worst misses** provide the evidence needed to revise a rubric.

If predictions have not been computed yet, use
[`ScoreDistributionAnalyzer`][trulens.benchmark.score_distribution.ScoreDistributionAnalyzer]
to execute the judge and diagnose score collapse, leniency bias, bimodal output,
and calibration:

```python
from trulens.benchmark.score_distribution import ScoreDistributionAnalyzer

distribution_report = ScoreDistributionAnalyzer(
    feedback_fn=provider.relevance,
    golden_set=development_set,
).run()
distribution_report.print_summary()
```

Score spread is necessary but not sufficient. A broad distribution can still be
systematically wrong, so compare it with human labels and `AlignmentReport`.

!!! warning "Correlation is not calibration"

    Rank correlation can be high while scores remain systematically too high or
    too low. Inspect MAE, Brier score, calibration bins, and threshold behavior
    alongside Spearman and Kendall.

    Pointwise scoring can also change with superficial prompt details such as
    score-label identifiers, rubric ordering, and reference-answer scores
    ([Li et al., 2025](https://arxiv.org/abs/2506.22316)). Treat these details as
    part of the frozen judge configuration.

## Compare paired results

Use [`CriteriaABTest`][trulens.benchmark.criteria_ab_test.CriteriaABTest] for two
prompt configurations while keeping the model, examples, temperature, and rows
fixed:

```python
from trulens.benchmark.criteria_ab_test import CriteriaABTest

criteria_report = CriteriaABTest(
    golden_set=development_set,
    variant_a={"fn": provider.relevance, "name": "default"},
    variant_b={
        "fn": provider.relevance,
        "name": "strict",
        "kwargs": {"criteria": strict_criteria},
    },
).run()
criteria_report.print_comparison()
```

A lower MAE identifies the closer variant, but a shifted mean score does not by
itself prove better alignment. Treat the reported score-shift permutation test
as evidence that variants score differently, then use ground-truth error,
ranking, and threshold metrics to decide whether the difference is useful.

Use
[`CrossModelAlignment`][trulens.benchmark.cross_model_alignment.CrossModelAlignment]
to run one feedback method across multiple providers or `model_engine` values:

```python
from trulens.benchmark.cross_model_alignment import CrossModelAlignment

model_report = CrossModelAlignment(
    judges=[
        {"provider": judge_a, "name": "judge-a"},
        {"provider": judge_b, "name": "judge-b"},
    ],
    feedback_method="relevance",
    golden_set=development_set,
).run()
model_report.print_matrix()
```

Pairwise agreement does not establish correctness. Include `expected_score`
labels and inspect `ground_truth_metrics()` before replacing a production judge.
Always run variants on the same rows and inspect skipped rows.

JudgeBench demonstrates why this distinction matters: strong judges can perform
near chance on challenging pairs with verifiable factual, logical, mathematical,
or coding answers ([Tan et al., 2024](https://arxiv.org/abs/2410.12784)). Use
human labels or objective references to establish correctness rather than using
inter-judge consensus as the target.

!!! warning "Compare identical row sets"

    Do not compare variants on different successfully scored subsets. If one
    variant fails on a row, exclude that row from every variant's paired result
    and report how many rows were skipped.

!!! warning "Revalidate decision thresholds"

    Do not keep a production threshold unchanged after changing `criteria`,
    `examples`, or `model_engine`. Better continuous alignment does not guarantee
    better pass/fail decisions at the old threshold.

For model comparisons, repeat trials when provider nondeterminism is material.
Latency and cost are operational criteria, not substitutes for alignment. See
the [model comparison cookbook](../../cookbook/models/openai/model_comparison_for_eval.ipynb)
for a repeated-trial example.

## Change one dimension: optimize or ensemble

Use [`FewShotOptimizer`][trulens.feedback.optimize.FewShotOptimizer] only with a
candidate pool and a separate labeled evaluation set:

```python
from trulens.feedback import FewShotOptimizer

optimizer = FewShotOptimizer(
    feedback_fn=provider.relevance,
    candidates=candidate_examples,
    eval_dataset=evaluation_rows,
    n_examples=3,
    metric="spearman",
    examples_format="structured",
)
result = optimizer.optimize()
```

Candidate scores use the judge's prompt scale, while evaluation labels use the
normalized `[0, 1]` provider return scale. Pass `result.best_examples` directly
to the provider method. After selection, rerun the complete judge on validation
and held-out data; the optimizer's objective is not the final report.

Iterative in-context approaches such as ALLURE improve evaluators by finding
large judge-human disagreements and adding selected failures as demonstrations
([Hasanbeig et al., 2023](https://arxiv.org/abs/2309.13701)). Because examples
selected from observed failures can overfit, keep `candidates`, `eval_dataset`,
and the final held-out rows disjoint.

Use [`Jury`][trulens.feedback.jury.Jury] when runtime reliability is worth the
added cost:

```python
from trulens.core import Metric
from trulens.feedback import Jury

jury = Jury(
    jurors=[judge_a, judge_b, judge_c],
    method="relevance_with_cot_reasons",
    aggregation="median",
)
metric = (
    Metric(implementation=jury, name="Jury Relevance").on_input().on_output()
)
```

`Jury` supports `mean`, `median`, `trimmed_mean`, `majority_vote`,
`weighted_mean`, and custom aggregation. It runs jurors in parallel, tolerates
individual failures, and returns the per-juror scores and reasons in metadata.
Cost scales with the number of jurors and latency follows the slowest juror.
A jury changes production scoring, while the alignment utilities diagnose
candidate judges. Benchmark both the aggregate score and each juror before
assigning weights.

Panels of smaller judges from disjoint model families have outperformed a single
large judge and reduced intra-model bias in several studied settings
([Verga et al., 2024](https://arxiv.org/abs/2404.18796)). Diversity is a design
hypothesis, not a guarantee: validate both the aggregate and individual jurors
against the same human labels before deploying the panel.

!!! warning "Limit claims to represented data"

    Do not claim one judge is best outside the domains, slices, languages, and
    score ranges represented by the labeled data. Treat model or jury selection
    as evidence for this evaluation target, not a universal leaderboard. A
    large-scale study found that judge-human alignment changes across evaluated
    tasks, properties, models, and annotator populations
    ([Bavaresco et al., 2024](https://arxiv.org/abs/2406.18403)).

## Confirm on held-out data

After selecting a configuration, rerun the same execution and diagnostic path on
`held_out_set` exactly once. Use the stable example IDs to align the persisted
judge outputs with human labels, then create the final `AlignmentReport`. Report
the complete configuration, thresholds, represented slices, and per-row outputs
alongside aggregate metrics so the decision can be reproduced and audited.

## Research references

- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685)
- [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment (Liu et al., 2023)](https://arxiv.org/abs/2303.16634)
- [Calibrating LLM-Based Evaluator (Liu et al., 2023)](https://arxiv.org/abs/2309.13308)
- [ALLURE: Auditing and Improving LLM-based Evaluation of Text using Iterative In-Context-Learning (Hasanbeig et al., 2023)](https://arxiv.org/abs/2309.13701)
- [Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models (Verga et al., 2024)](https://arxiv.org/abs/2404.18796)
- [LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks (Bavaresco et al., 2024)](https://arxiv.org/abs/2406.18403)
- [JudgeBench: A Benchmark for Evaluating LLM-based Judges (Tan et al., 2024)](https://arxiv.org/abs/2410.12784)
- [Evaluating Scoring Bias in LLM-as-a-Judge (Li et al., 2025)](https://arxiv.org/abs/2506.22316)
