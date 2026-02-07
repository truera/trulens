from collections import defaultdict
from datetime import datetime
import json
import os
import time

import dspy
from dspy.teleprompt import GEPA
import pandas as pd
from trulens.feedback.v2 import feedback as feedback_v2

## set ENV variables
os.environ["SNOWFLAKE_ACCOUNT_ID"] = "SFCOGSOPS-SNOWHOUSE-AWS-US-WEST-2"
# os.environ["SNOWFLAKE_USER"] = "AJIA"
os.environ["SNOWFLAKE_JWT"] = "pat/..."
lm = dspy.LM(
    "snowflake/claude-sonnet-4-5",
    temperature=1,
    max_tokens=32000,
    api_key=os.environ["SNOWFLAKE_JWT"],
)
dspy.configure(lm=lm)

# Setup logging to file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = open(f"gepa_gaia_run_{timestamp}.log", "w", encoding="utf-8")


def log(message=""):
    """Write message to log file and optionally to console."""
    # print(message, file=log_file, flush=True)


def truncate_trace(trace, max_chars=80000):
    """Truncate trace if it exceeds max_chars to avoid token limits."""
    if len(trace) > max_chars:
        return trace[:max_chars] + "\n\n[... trace truncated due to length ...]"
    return trace


df = pd.read_csv(
    "examples/experimental/GPA Judge Error Analysis - TRAIN_CSV.csv"
)
judge_df = pd.read_csv(
    "examples/experimental/TRAIL_GAIA_Judge_Output_Per_Trace.csv"
)
test_df = pd.read_csv(
    "examples/experimental/GPA Judge Error Analysis - TEST_CSV.csv"
)

for i in range(len(df)):
    file_name = df.iloc[i]["Filename"]
    df.loc[i, "trace"] = open(
        f"examples/experimental/{file_name}.txt",
        "r",
    ).read()

# Prepare test_df similarly
for i in range(len(test_df)):
    file_name = test_df.iloc[i]["Filename"]
    test_df.loc[i, "trace"] = open(
        f"examples/experimental/{file_name}.txt",
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


df["gpa_list"] = df["Align_Judges"].apply(parse_gpa_category)
df["caught_list"] = df["Caught"].apply(parse_gpa_category)
df["aligned_list"] = df["Align_Judges"].apply(parse_gpa_category)

# Parse test_df similarly
test_df["gpa_list"] = test_df["Align_Judges"].apply(parse_gpa_category)
test_df["caught_list"] = test_df["Caught"].apply(parse_gpa_category)
test_df["aligned_list"] = test_df["Align_Judges"].apply(parse_gpa_category)

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

for file_value, file_group in df.groupby("Filename"):
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
            if (
                "Raw Error" in df.columns
                and pd.notna(df.loc[idx, "Raw Error"])
                and str(df.loc[idx, "Raw Error"]).strip() != ""
            ):
                errors_list.append(df.loc[idx, "Raw Error"])

        # Create dspy.Example directly and add to category group
        example = dspy.Example(
            trace=open(
                f"examples/experimental/{file_value}.txt",
                "r",
            ).read(),
            error_category=error_code,
            errors=errors_list,
            file=file_value,  # Keep for debugging
        ).with_inputs("trace")

        grouped_examples[error_code].append(example)

# ============================================================
# BUILD TEST EXAMPLES GROUPED BY ERROR CATEGORY
# ============================================================

test_grouped_examples = defaultdict(list)

for file_value, file_group in test_df.groupby("Filename"):
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
            if (
                "Raw Error" in test_df.columns
                and pd.notna(test_df.loc[idx, "Raw Error"])
                and str(test_df.loc[idx, "Raw Error"]).strip() != ""
            ):
                errors_list.append(test_df.loc[idx, "Raw Error"])

        # Create dspy.Example directly and add to category group
        example = dspy.Example(
            trace=open(
                f"examples/experimental/{file_value}.txt",
                "r",
            ).read(),
            error_category=error_code,
            errors=errors_list,
            file=file_value,  # Keep for debugging
        ).with_inputs("trace")

        test_grouped_examples[error_code].append(example)


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

    # be more specific about what to produce in feedback analysis
    # feedback_analysis: str = dspy.OutputField(
    #     desc="Detailed analysis: list each golden error and identify whether the student_critique caught or missed it. The student_critique does not need to explicitly mention the golden error category, but it should match the description and evidence of the golden error."
    # )
    feedback_analysis: str = dspy.OutputField(
        desc="Detailed analysis: which **golden errors** were caught versus missed. **A golden error is considered 'caught' if the student_critique captures its essential nature or core problem**, even if the language or specific error category doesn't match the golden error exactly. If the student_critique mentions a relevant error, count it."
    )
    overall_score: str = dspy.OutputField(
        desc="Recall: fraction of golden errors successfully identified. Return a string in the format: # of caught / # of golden errors (e.g. 3/5)."
    )


meta_judge = dspy.ChainOfThought(MetaJudgeSignature)
# print history of Chain of Thought module (inspect_history)
# print(meta_judge.inspect_history())
# print(f"meta judge reasoning: {meta_judge['reasoning']}")

# ============================================================
# 4. META JUDGE VALIDATION
# ============================================================

validation_results = "examples/experimental/lax_validation_results.txt"


def meta_judge_validation():
    correct_count = 0
    total_actual_aligned = 0
    total_actual_caught = 0
    total_metajudge_aligned = 0
    total_metajudge_caught = 0
    for category, category_examples in grouped_examples.items():
        example_count = 0
        for ex in category_examples:
            if example_count >= 15:
                break
            example_count += 1
            filename = ex.file
            # print(f"filename: {filename}")
            judge_output = judge_df[judge_df["filename"] == filename][
                category
            ].values[0]
            # print(f"judge_output: {judge_output}")
            evaluation = meta_judge(
                agent_trace=ex.trace,
                golden_errors=ex.errors,
                student_critique=judge_output,
            )
            score = evaluation.overall_score.split("/")
            total_metajudge_caught += int(score[0])
            total_metajudge_aligned += int(score[1])
            # print(f"evaluation: {evaluation}")
            # print(f"aligned_list: {df[df['Filename'] == filename]['aligned_list']}")
            num_aligned = 0
            num_caught = 0
            for _, row in df[df["Filename"] == filename].iterrows():
                if category in row["aligned_list"]:
                    num_aligned += 1
                    if category in row["caught_list"]:
                        num_caught += 1
            print(f"true caught/aligned ratio: {num_caught / num_aligned}")
            print(f"metajudge score: {score}")
            if num_caught / num_aligned == int(score[0]) / int(score[1]):
                correct_count += 1
            total_actual_aligned += num_aligned
            total_actual_caught += num_caught
            with open(validation_results, "a") as f:
                f.write(f"filename: {filename}\n")
                f.write(f"category: {category}\n")
                f.write(f"evaluation: {evaluation}\n")
                f.write(
                    f"metajudge caught/aligned ratio: {int(score[0])}/{int(score[1])}\n"
                )
                f.write(
                    f"true caught/aligned ratio: {num_caught}/{num_aligned}\n"
                )
                f.write(
                    f"correct: {num_caught / num_aligned == int(score[0]) / int(score[1])}\n"
                )
                f.write("\n")
    with open(validation_results, "a") as f:
        f.write(f"correct count: {correct_count}\n")
        f.write("total count: 60\n")
        f.write(f"accuracy: {correct_count / 60}\n")
        f.write(f"total_actual_aligned: {total_actual_aligned}\n")
        f.write(f"total_actual_caught: {total_actual_caught}\n")
        f.write(f"total_metajudge_aligned: {total_metajudge_aligned}\n")
        f.write(f"total_metajudge_caught: {total_metajudge_caught}\n")
        f.write(
            f"total_actual_caught/total_actual_aligned: {total_actual_caught / total_actual_aligned}\n"
        )
        f.write(
            f"total_metajudge_caught/total_metajudge_aligned: {total_metajudge_caught / total_metajudge_aligned}\n"
        )


# meta_judge_validation()

# ============================================================
# 4. METRIC FUNCTION (uses meta-judge to score)
# ============================================================
#
# Returns dspy.Prediction with 'score' and 'feedback' fields.
# GEPA uses the 'score' for optimization and the 'feedback' to guide
# prompt improvements.


def gepa_metric_with_feedback(
    gold, pred, trace=None, pred_name=None, pred_trace=None, category=None
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

    # Truncate trace to avoid token limits
    truncated_agent_trace = truncate_trace(gold.trace)

    # Ask meta-judge to evaluate
    evaluation = meta_judge(
        agent_trace=truncated_agent_trace,
        golden_errors=golden_errors_str,
        student_critique=pred.critique,
    )

    numeric_score = float(evaluation.overall_score.split("/")[0]) / float(
        evaluation.overall_score.split("/")[1]
    )
    return dspy.Prediction(
        score=numeric_score,
        feedback=evaluation.feedback_analysis,
    )


# ============================================================
# 5. RUN GEPA OPTIMIZATION PER CATEGORY
# ============================================================

optimized_prompts = {}
optimized_students = {}

output_file = "gaia_optimized_judge_prompts_plan_adherence.json"

for category, category_examples in grouped_examples.items():
    if category in ["LC", "EE", "TC", "TS", "PQ"]:
        continue
    log(f"\n{'=' * 60}")
    log(f"OPTIMIZING CATEGORY: {category}")
    log(f"{'=' * 60}")

    # A. Get the starting prompt from your TruLens feedback system
    feedback_class_name = error_feedback_mapping[category]
    starting_instruction = getattr(
        feedback_v2, feedback_class_name
    ).system_prompt

    print("student is being created")
    # B. Create a dspy.Predict instance
    student = dspy.Predict(StudentJudgeSignature)

    # C. Inject your existing TruLens prompt as the starting point
    student.signature = student.signature.with_instructions(
        starting_instruction
    )

    # D. Test BEFORE optimization
    log("\n--- Testing BEFORE optimization ---")
    test_ex = category_examples[0]

    print("baseline prediction before")
    print(f"Test trace length: {len(test_ex.trace)}")

    # Truncate trace if too large (keep first 80k chars ≈ 20k tokens)
    max_trace_chars = 80000
    truncated_trace = test_ex.trace
    if len(test_ex.trace) > max_trace_chars:
        truncated_trace = (
            test_ex.trace[:max_trace_chars]
            + "\n\n[... trace truncated due to length ...]"
        )
        print(
            f"⚠️  Trace truncated from {len(test_ex.trace)} to {len(truncated_trace)} chars"
        )

    print(
        f"Using trace length: {len(truncated_trace)} chars (~{len(truncated_trace) / 4:.0f} tokens)"
    )

    try:
        baseline_pred = student(trace=truncated_trace)
        print("baseline prediction after")
    except Exception as e:
        print(f"ERROR during baseline prediction: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        raise

    baseline_result = gepa_metric_with_feedback(
        gold=test_ex, pred=baseline_pred
    )
    log(f"\nBaseline score: {baseline_result.score:.2f}")
    log(f"Baseline feedback: {baseline_result.feedback}...")
    time.sleep(1)  # Rate limiting: wait before starting optimization

    # E. Split into train/val (for small datasets, use all for both)
    if len(category_examples) >= 10:
        split_idx = int(len(category_examples) * 0.8)
        train_set = category_examples[:split_idx]
        val_set = category_examples[split_idx:]
    else:
        # With <10 examples, use all for both train and val
        log(
            f"  ⚠️  Small dataset ({len(category_examples)} examples) - using all for train & val"
        )
        train_set = category_examples
        val_set = category_examples

    log(f"\nTrain: {len(train_set)} examples, Val: {len(val_set)} examples")

    # F. Run GEPA optimization (conservative settings for small dataset)
    log("\n--- Running GEPA optimization ---")
    optimizer = GEPA(
        metric=gepa_metric_with_feedback,
        # Use a stronger model for optimization/reflection (recommended)
        reflection_lm=dspy.LM(
            "snowflake/claude-sonnet-4-5",
            temperature=1,
            max_tokens=32000,
            api_key=os.environ["SNOWFLAKE_JWT"],
        ),
        # Budget control - CRITICAL for small datasets!
        auto="medium",  # For testing.
        # Tracking (useful for debugging)
        track_stats=True,
        track_best_outputs=True,
        # Parallelization - set to 1 to avoid rate limiting
        num_threads=1,  # Sequential processing to avoid rate limits
    )

    optimized_student = optimizer.compile(
        student=student, trainset=train_set, valset=val_set
    )

    # G. Test AFTER optimization on VALIDATION SET
    log("\n--- Testing AFTER optimization (on validation set) ---")
    log(
        f"\nEvaluating optimized judge on {len(val_set)} validation traces for {category}:"
    )

    all_scores_before = []
    all_scores_after = []

    for i, ex in enumerate(val_set):
        truncated = truncate_trace(ex.trace)

        # Test unoptimized
        baseline_pred_ex = student(trace=truncated)
        result_before = gepa_metric_with_feedback(
            gold=ex, pred=baseline_pred_ex
        )
        score_before = result_before.score
        all_scores_before.append(score_before)

        # Test optimized
        optimized_pred_ex = optimized_student(trace=truncated)
        result_after = gepa_metric_with_feedback(
            gold=ex, pred=optimized_pred_ex
        )
        score_after = result_after.score
        all_scores_after.append(score_after)

        log(
            f"  Trace {i + 1} ({ex['file']}): {score_before:.2f} → {score_after:.2f} ({score_after - score_before:+.2f})"
        )
        time.sleep(
            1
        )  # Rate limiting: wait 1 second between validation examples

    avg_before = sum(all_scores_before) / len(all_scores_before)
    avg_after = sum(all_scores_after) / len(all_scores_after)

    log(
        f"\n  📊 Average Score: {avg_before:.2f} → {avg_after:.2f} ({avg_after - avg_before:+.2f})"
    )

    # H. Save the optimized prompt
    final_prompt = optimized_student.signature.instructions
    optimized_prompts[category] = final_prompt
    optimized_students[category] = optimized_student

    with open(output_file, "w") as f:
        json.dump(optimized_prompts[category], f, indent=2)

    log("\n--- Optimized prompt ---")
    log(final_prompt)

# ============================================================
# 7. FINAL EVALUATION ON TEST SET
# ============================================================

log(f"\n{'=' * 60}")
log("FINAL EVALUATION ON TEST SET")
log(f"{'=' * 60}\n")

for category, optimized_student in optimized_students.items():
    test_examples = test_grouped_examples.get(category, [])
    if not test_examples:
        log(f"{category}: No test examples")
        continue

    scores = []

    for ex in test_examples:
        log(f"trace: {ex.file}")
        log(f"errors: {ex.errors}")
        truncated = truncate_trace(ex.trace)
        pred = optimized_student(trace=truncated)
        log(f"pred: {pred}")
        result = gepa_metric_with_feedback(gold=ex, pred=pred)
        log(f"result: {result}")
        scores.append(result.score)
        log("********************************************************")
        time.sleep(1)  # Rate limiting: wait 1 second between test examples

    avg_score = sum(scores) / len(scores)
    log(f"{category}: Test Avg Score = {avg_score:.2f} (n={len(scores)})")


# Close the log file
log_file.close()
