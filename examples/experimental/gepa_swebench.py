from collections import defaultdict
import json

import dspy
from dspy.teleprompt import GEPA
import pandas as pd
from trulens.feedback.v2 import feedback as feedback_v2

api_key = "..."
lm = dspy.LM("openai/gpt-4.1-mini", temperature=1, api_key=api_key)
dspy.configure(lm=lm)

df = pd.read_csv("train_trail_annotations.csv")

for i in range(len(df)):
    file_name = df.iloc[i]["file"]
    df.loc[i, "trace"] = open(
        f"examples/experimental/SWE_Bench/{file_name.replace('.json', '.txt')}",
        "r",
    ).read()


# Convert GPA Category (AJ) to gpa_list, ignoring empty values
def parse_gpa_category(value):
    """Parse GPA Category (AJ) column to list of error codes, ignoring empty values."""
    # Check if value is null/empty - if so, ignore it
    if (
        pd.isna(value)
        or value == ""
        or (isinstance(value, str) and value.strip() == "")
    ):
        return []

    # Parse the string value
    if isinstance(value, str):
        # Handle comma-separated values like "LC, EE" or "LC,EE"
        codes = [code.strip() for code in value.split(",")]
        # Filter out empty strings and return
        return [code for code in codes if code]

    return []


df["gpa_list"] = df["GPA Category (AJ)"].apply(parse_gpa_category)

# ============================================================
# BUILD EXAMPLES GROUPED BY ERROR CATEGORY
# ============================================================

error_feedback_mapping = {
    "LC": "LogicalConsistency",
    "EE": "ExecutionEfficiency",
    "PA": "PlanAdherence",
    "PQ": "PlanQuality",
    "TC": "ToolCalling",
    "TS": "ToolSelection",
}

# Build grouped structure directly
grouped_examples = defaultdict(list)

for file_value, file_group in df.groupby("file"):
    # Get all unique error codes in this trace
    all_error_codes = set()
    for gpa_list in file_group["gpa_list"]:
        all_error_codes.update(gpa_list)

    # For each error code, create a dspy.Example and add to the category group
    for error_code in all_error_codes:
        rows_with_error = file_group[
            file_group["gpa_list"].apply(lambda x: error_code in x)
        ]

        # Collect all errors from these rows
        errors_list = []
        for idx in rows_with_error.index:
            if "error" in df.columns and pd.notna(df.loc[idx, "error"]):
                errors_list.append(df.loc[idx, "error"])

        # Create dspy.Example directly and add to category group
        example = dspy.Example(
            trace=file_group["trace"].iloc[0],
            error_category=error_code,
            errors=errors_list,
            file=file_value,  # Keep for debugging
        ).with_inputs("trace")

        grouped_examples[error_code].append(example)

# Print detailed summary showing different errors per trace
print("\n" + "=" * 60)
print("EXAMPLES GROUPED BY CATEGORY")
print("=" * 60)
total_examples = 0

for category, category_examples in grouped_examples.items():
    print(f"\n{category} ({len(category_examples)} examples):")
    print("-" * 40)

    for i, ex in enumerate(category_examples):
        trace_preview = ex["trace"][:100].replace("\n", " ")
        print(f"  Example {i + 1}:")
        print(f"    File: {ex['file']}")
        print(f"    Trace: {trace_preview}... ({len(ex['trace'])} chars)")
        print(f"    Golden Errors for {category}: {ex['errors']}")

    total_examples += len(category_examples)

print(f"{'=' * 60}")
print(
    f"Total: {total_examples} examples across {len(grouped_examples)} categories"
)
print(f"{'=' * 60}\n")

# ============================================================
# 2. STUDENT JUDGE SIGNATURE (Fixed structure)
# ============================================================


class StudentJudgeSignature(dspy.Signature):
    """Analyze an agent trace for errors based on a specific error category."""

    trace: str = dspy.InputField(desc="The raw execution trace of the agent.")

    critique: str = dspy.OutputField(
        desc="A detailed critique listing all errors found in the trace given the criteria for this category."
    )


# ============================================================
# 3. META-JUDGE (evaluates the student judge)
# ============================================================


# BE MORE SPECIFIC ABOUT EACH SIGNATURE
class MetaJudgeSignature(dspy.Signature):
    """
    Evaluate how well a student judge identified errors.
    Compare the student's critique against golden/ground-truth errors.
    Return a score (0.0-1.0) and detailed feedback.
    """

    agent_trace: str = dspy.InputField(
        desc="The raw agent trace that was judged."
    )
    golden_errors: str = dspy.InputField(
        desc="Ground truth errors that SHOULD be found (newline-separated list)."
    )
    student_critique: str = dspy.InputField(
        desc="The student judge's critique/output."
    )

    feedback_analysis: str = dspy.OutputField(
        desc="Detailed analysis: which golden errors were caught vs missed, and any false positives."
    )
    recall_score: float = dspy.OutputField(
        desc="Recall: fraction of golden errors successfully identified (0.0 to 1.0)."
    )
    overall_score: float = dspy.OutputField(
        desc="Overall quality score considering recall and precision (0.0 to 1.0)."
    )


meta_judge = dspy.ChainOfThought(MetaJudgeSignature)

# ============================================================
# 4. METRIC FUNCTION (uses meta-judge to score)
# ============================================================
#
# Returns dspy.Prediction with 'score' and 'feedback' fields.
# GEPA uses the 'score' for optimization and the 'feedback' to guide
# prompt improvements.


def gepa_metric_with_feedback(
    gold, pred, trace=None, pred_name=None, pred_trace=None
):
    """
    Metric for GEPA optimization.
    GEPA requires this EXACT signature: (gold, pred, trace, pred_name, pred_trace)

    Args:
        gold: dspy.Example with .trace, .error_category, .errors (golden/ground truth)
        pred: The student judge's output with .critique
        trace: DSPy's internal execution trace (optional, usually ignore)
        pred_name: Name of the prediction field (optional, usually ignore)
        pred_trace: Trace of the prediction (optional, usually ignore)

    Returns:
        dspy.Prediction with 'score' and 'feedback' fields
    """
    # Format golden errors nicely
    golden_errors_str = "\n".join([f"- {err}" for err in gold.errors])

    # Ask meta-judge to evaluate
    evaluation = meta_judge(
        agent_trace=gold.trace,
        golden_errors=golden_errors_str,
        student_critique=pred.critique,
    )

    return dspy.Prediction(
        score=float(evaluation.overall_score),
        feedback=evaluation.feedback_analysis,
    )


# ============================================================
# 5. RUN GEPA OPTIMIZATION PER CATEGORY
# ============================================================

optimized_prompts = {}
optimized_students = {}

for category, category_examples in grouped_examples.items():
    print(f"\n{'=' * 60}")
    print(f"OPTIMIZING CATEGORY: {category}")
    print(f"{'=' * 60}")

    # A. Get the starting prompt from your TruLens feedback system
    feedback_class_name = error_feedback_mapping[category]
    starting_instruction = getattr(
        feedback_v2, feedback_class_name
    ).system_prompt

    print(starting_instruction + "...")

    # B. Create a dspy.Predict instance
    student = dspy.Predict(StudentJudgeSignature)

    # C. Inject your existing TruLens prompt as the starting point
    student.signature = student.signature.with_instructions(
        starting_instruction
    )

    # D. Test BEFORE optimization
    print("\n--- Testing BEFORE optimization ---")
    test_ex = category_examples[0]

    print("\n[DEBUGGING] What the LLM receives:")
    print(
        f"  Instruction (first 200 chars): {student.signature.instructions[:200]}..."
    )
    print(f"  Input trace (first 200 chars): {test_ex.trace[:200]}...")
    print("  Expected to output: 'critique' field")

    baseline_pred = student(trace=test_ex.trace)

    print("\n[DEBUGGING] What the LLM returns:")
    print(f"  Type: {type(baseline_pred)}")
    print(
        f"  Fields: {baseline_pred.keys() if hasattr(baseline_pred, 'keys') else 'N/A'}"
    )
    print(f"  Critique : {baseline_pred.critique}...")

    baseline_result = gepa_metric_with_feedback(
        gold=test_ex, pred=baseline_pred
    )
    print(f"\nBaseline score: {baseline_result.score:.2f}")
    print(f"Baseline feedback: {baseline_result.feedback}...")

    # E. Split into train/val (for small datasets, use all for both)
    if len(category_examples) >= 10:
        split_idx = int(len(category_examples) * 0.8)
        train_set = category_examples[:split_idx]
        val_set = category_examples[split_idx:]
    else:
        # With <10 examples, use all for both train and val
        print(
            f"  ⚠️  Small dataset ({len(category_examples)} examples) - using all for train & val"
        )
        train_set = category_examples
        val_set = category_examples

    print(f"\nTrain: {len(train_set)} examples, Val: {len(val_set)} examples")

    # F. Run GEPA optimization (conservative settings for small dataset)
    print("\n--- Running GEPA optimization ---")
    optimizer = GEPA(
        metric=gepa_metric_with_feedback,
        # Use a stronger model for optimization/reflection (recommended)
        reflection_lm=dspy.LM("openai/gpt-5", api_key=api_key),
        # Budget control - CRITICAL for small datasets!
        # "max_full_evals" = how many complete trainset evaluations GEPA can do
        # With 4-5 examples per category, start conservative:
        max_full_evals=2,  # For testing. Increase to 5-10 when you have more data
        # Tracking (useful for debugging)
        track_stats=True,
        track_best_outputs=True,
        # Parallelization (won't help much with small dataset, but doesn't hurt)
        num_threads=4,  # Keep low for small datasets
    )

    optimized_student = optimizer.compile(
        student=student, trainset=train_set, valset=val_set
    )

    # G. Test AFTER optimization across ALL traces in this category
    print("\n--- Testing AFTER optimization ---")
    print(
        f"\nEvaluating optimized judge on all {len(category_examples)} traces for {category}:"
    )

    all_scores_before = []
    all_scores_after = []

    for i, ex in enumerate(category_examples):
        # Test unoptimized
        baseline_pred_ex = student(trace=ex.trace)
        result_before = gepa_metric_with_feedback(
            gold=ex, pred=baseline_pred_ex
        )
        score_before = result_before.score
        all_scores_before.append(score_before)

        # Test optimized
        optimized_pred_ex = optimized_student(trace=ex.trace)
        result_after = gepa_metric_with_feedback(
            gold=ex, pred=optimized_pred_ex
        )
        score_after = result_after.score
        all_scores_after.append(score_after)

        print(
            f"  Trace {i + 1} ({ex['file']}): {score_before:.2f} → {score_after:.2f} ({score_after - score_before:+.2f})"
        )

    avg_before = sum(all_scores_before) / len(all_scores_before)
    avg_after = sum(all_scores_after) / len(all_scores_after)

    print(
        f"\n  📊 Average Score: {avg_before:.2f} → {avg_after:.2f} ({avg_after - avg_before:+.2f})"
    )
    print(
        f"  🎯 The {category} judge now works across ALL these different traces!"
    )

    # H. Save the optimized prompt
    final_prompt = optimized_student.signature.instructions
    optimized_prompts[category] = final_prompt
    optimized_students[category] = optimized_student

    print("\n--- Optimized prompt (first 300 chars) ---")
    print(final_prompt[:300] + "...")

# ============================================================
# 6. FINAL EVALUATION ACROSS ALL CATEGORIES
# ============================================================

print(f"\n{'=' * 60}")
print("FINAL EVALUATION ACROSS ALL CATEGORIES")
print(f"{'=' * 60}\n")

for category, optimized_student in optimized_students.items():
    category_examples = grouped_examples[category]
    scores = []

    for ex in category_examples:
        pred = optimized_student(trace=ex.trace)
        result = gepa_metric_with_feedback(gold=ex, pred=pred)
        scores.append(result.score)

    avg_score = sum(scores) / len(scores)
    print(f"{category}: Avg Score = {avg_score:.2f} (n={len(scores)})")

# ============================================================
# 7. SAVE OPTIMIZED PROMPTS
# ============================================================

print(f"\n{'=' * 60}")
print("SAVING OPTIMIZED PROMPTS")
print(f"{'=' * 60}\n")

output_file = "optimized_judge_prompts.json"
with open(output_file, "w") as f:
    json.dump(optimized_prompts, f, indent=2)

print(f"Saved optimized prompts to: {output_file}")
print("\nYou can now use these prompts in your production judge system!")
