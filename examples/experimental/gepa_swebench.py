"""
GEPA Judge Optimization and Evaluation Script for SWE-Bench

USAGE:
------
1. To run GEPA optimization (generate new prompts):
   - Set RUN_OPTIMIZATION = True (around line 406)
   - Optionally adjust categories_to_skip in the categories_to_optimize line (around line 694)
   - Run the script

2. To evaluate using saved prompts (skip optimization):
   - Set RUN_OPTIMIZATION = False (around line 406)
   - Set prompt_file to your saved prompts file (around line 737)
   - Run the script - it will load prompts and run test evaluation only
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import wraps
import json
import os
import threading
import time

import dspy
from dspy.teleprompt import GEPA
import pandas as pd
from trulens.feedback.v2 import feedback as feedback_v2

## set ENV variables
os.environ["SNOWFLAKE_ACCOUNT_ID"] = "SFCOGSOPS-SNOWHOUSE-AWS-US-WEST-2"
# os.environ["SNOWFLAKE_USER"] = "AJIA"
os.environ["SNOWFLAKE_JWT"] = "pat/"


# ============================================================
# RETRY LOGIC FOR RATE LIMITS (429 errors)
# ============================================================


def retry_with_backoff(max_retries=5, initial_delay=2):
    """Decorator to retry API calls with exponential backoff on 429 errors."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "rate limit" in error_str.lower():
                        if attempt < max_retries - 1:
                            delay = initial_delay * (2**attempt)
                            print(
                                f"⚠️  Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {delay}s..."
                            )
                            time.sleep(delay)
                            continue
                    raise
            raise Exception(f"Failed after {max_retries} retries")

        return wrapper

    return decorator


# Create LM with retry logic
lm = dspy.LM(
    "snowflake/claude-sonnet-4-5",
    temperature=1,
    max_tokens=32000,
    api_key=os.environ["SNOWFLAKE_JWT"],
)

# Wrap LM calls with retry logic for rate limit handling
original_call = lm.__call__
lm.__call__ = retry_with_backoff(max_retries=5, initial_delay=2)(original_call)

dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
dspy.configure(lm=lm)

# Setup logging to file with thread-safe lock
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = open(
    f"auto-light_swebench_run4_{timestamp}.log", "w", encoding="utf-8"
)
log_lock = threading.Lock()  # Thread-safe logging


def log(message=""):
    """Write message to log file in a thread-safe manner."""
    with log_lock:
        print(message, file=log_file, flush=True)


df = pd.read_csv("examples/experimental/SWE-Bench_Train.csv")
test_df = pd.read_csv("examples/experimental/SWE-Bench_Test.csv")

# for i in range(len(df)):
#     file_name = df.iloc[i]["file"]
#     df.loc[i, "trace"] = open(
#         f"examples/experimental/SWE_Bench/{file_name}".replace("json", "txt"),
#         "r",
#     ).read()

# # Prepare test_df similarly
# for i in range(len(test_df)):
#     file_name = test_df.iloc[i]["file"]
#     test_df.loc[i, "trace"] = open(
#         f"examples/experimental/SWE_Bench/{file_name}".replace("json", "txt"),
#         "r",
#     ).read()


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

# Parse test_df similarly
test_df["gpa_list"] = test_df["GPA Category (AJ)"].apply(parse_gpa_category)

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
            if (
                "error" in df.columns
                and pd.notna(df.loc[idx, "error"])
                and str(df.loc[idx, "error"]).strip() != ""
            ):
                errors_list.append(df.loc[idx, "error"])

        # Create dspy.Example directly and add to category group
        example = dspy.Example(
            trace=open(
                f"examples/experimental/SWE_Bench/{file_value}".replace(
                    "json", "txt"
                ),
                "r",
            ).read(),
            error_category=error_code,
            errors=errors_list,
            file=file_value,  # Keep for debugging
        ).with_inputs("trace")

        grouped_examples[error_code].append(example)

# ============================================================
# PRINT EXAMPLES OF GROUPED_EXAMPLES
# ============================================================

# print(f"\n{'=' * 60}")
# print("GROUPED EXAMPLES STRUCTURE")
# print(f"{'=' * 60}\n")

# print(f"Total categories: {len(grouped_examples)}")
# print(f"Categories: {list(grouped_examples.keys())}\n")

# for category, examples in grouped_examples.items():
#     print(f"\n{'─' * 60}")
#     print(f"Category: {category} ({error_feedback_mapping.get(category, 'Unknown')})")
#     print(f"Number of examples: {len(examples)}")
#     print(f"{'─' * 60}")

#     # Print first example details
#     if examples:
#         ex = examples[0]
#         print(f"\nExample 1:")
#         print(f"  File: {ex.file}")
#         print(f"  Error category: {ex.error_category}")
#         print(f"  Number of errors: {len(ex.errors)}")
#         print(f"  Errors:")
#         for i, error in enumerate(ex.errors, 1):
#             print(f"    {i}. {error[:200]}{'...' if len(error) > 200 else ''}")
#         print(f"  Trace length: {len(ex.trace)} chars (~{len(ex.trace) / 4:.0f} tokens)")
#         print(f"  Trace preview (first 300 chars):")
#         print(f"    {ex.trace[:300]}...")

#     # Print second example if exists
#     if len(examples) > 1:
#         ex = examples[1]
#         print(f"\nExample 2:")
#         print(f"  File: {ex.file}")
#         print(f"  Error category: {ex.error_category}")
#         print(f"  Number of errors: {len(ex.errors)}")
#         print(f"  Errors:")
#         for i, error in enumerate(ex.errors, 1):
#             print(f"    {i}. {error[:200]}{'...' if len(error) > 200 else ''}")
#         print(f"  Trace length: {len(ex.trace)} chars (~{len(ex.trace) / 4:.0f} tokens)")

# print(f"\n{'=' * 60}\n")

# ============================================================
# BUILD TEST EXAMPLES GROUPED BY ERROR CATEGORY
# ============================================================

test_grouped_examples = defaultdict(list)

for file_value, file_group in test_df.groupby("file"):
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
                "error" in test_df.columns
                and pd.notna(test_df.loc[idx, "error"])
                and str(test_df.loc[idx, "error"]).strip() != ""
            ):
                errors_list.append(test_df.loc[idx, "error"])

        # Create dspy.Example directly and add to category group
        example = dspy.Example(
            trace=open(
                f"examples/experimental/SWE_Bench/{file_value}".replace(
                    "json", "txt"
                ),
                "r",
            ).read(),
            error_category=error_code,
            errors=errors_list,
            file=file_value,  # Keep for debugging
        ).with_inputs("trace")

        test_grouped_examples[error_code].append(example)


# ============================================================
# PRINT EXAMPLES OF TEST_GROUPED_EXAMPLES
# ============================================================

# print(f"\n{'=' * 60}")
# print("TEST GROUPED EXAMPLES STRUCTURE")
# print(f"{'=' * 60}\n")

# print(f"Total categories: {len(test_grouped_examples)}")
# print(f"Categories: {list(test_grouped_examples.keys())}\n")

# for category, examples in test_grouped_examples.items():
#     print(f"\n{'─' * 60}")
#     print(f"Category: {category} ({error_feedback_mapping.get(category, 'Unknown')})")
#     print(f"Number of test examples: {len(examples)}")
#     print(f"{'─' * 60}")

#     # Print first example details
#     if examples:
#         ex = examples[0]
#         print(f"\nTest Example 1:")
#         print(f"  File: {ex.file}")
#         print(f"  Error category: {ex.error_category}")
#         print(f"  Number of errors: {len(ex.errors)}")
#         print(f"  Errors:")
#         for i, error in enumerate(ex.errors, 1):
#             print(f"    {i}. {error[:200]}{'...' if len(error) > 200 else ''}")
#         print(f"  Trace length: {len(ex.trace)} chars (~{len(ex.trace) / 4:.0f} tokens)")

# print(f"\n{'=' * 60}\n")

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
        desc="""Detailed analysis of which **golden errors** were CAUGHT (identified/mentioned) versus MISSED (not mentioned at all).

Evaluation criteria for 'caught':
- A golden error is considered 'caught' if the critique mentions or describes the same underlying problem behavior in ANY way.
- Exact wording, impact labels, specific error categories, or location IDs do not need to match, as the critique does not have access to the golden errors.
- The critique does not need to explicitly characterize the impact, error category, or location of the golden error.
- Focus on: Does the critique correctly identify the error behavior?

For each golden error, state:
1. Whether it was CAUGHT or MISSED
2. If CAUGHT: Quote the relevant portion of the critique that mentions it
3. If MISSED: Explain what the critique missed and provide specific guidance for improvement (e.g., "Should have identified the repeated failed tool calls with invalid arguments" or "Missed the inefficient search pattern - look for unnecessary repeated operations")
"""
    )
    overall_score: str = dspy.OutputField(
        desc="Recall score showing fraction of golden errors successfully caught. Return ONLY a simple fraction string in the format: <integer caught>/<integer total> (e.g., '3/5' or '0/4'). Each error is either fully caught or fully missed - no partial credit. Count conservatively but fairly based on your analysis above."
    )


meta_judge = dspy.ChainOfThought(MetaJudgeSignature)
# print history of Chain of Thought module (inspect_history)
# print(meta_judge.inspect_history())
# print(f"meta judge reasoning: {meta_judge['reasoning']}")


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
    try:
        # Check if prediction has the required field
        if not hasattr(pred, "critique"):
            print(
                "⚠️  Prediction missing 'critique' field - skipping this example"
            )
            print(f"   Prediction fields: {list(pred.__dict__.keys())}")
            return dspy.Prediction(
                score=0.0, feedback="Missing critique field - example skipped"
            )

        # Format golden errors nicely
        golden_errors_str = "\n".join([f"- {err}" for err in gold.errors])

        # Ask meta-judge to evaluate
        try:
            evaluation = meta_judge(
                agent_trace=gold.trace,
                golden_errors=golden_errors_str,
                student_critique=pred.critique,
            )
        except Exception as e:
            print(
                f"⚠️  Meta-judge failed - skipping this example: {type(e).__name__}: {str(e)[:100]}"
            )
            return dspy.Prediction(
                score=0.0, feedback="Meta-judge error - example skipped"
            )

        # Parse the score safely - extract just the fraction part (e.g., "4/5")
        try:
            # Get only the first line and strip whitespace
            score_line = evaluation.overall_score.strip().split("\n")[0].strip()
            score_parts = score_line.split("/")
            numeric_score = float(score_parts[0]) / float(score_parts[1])
        except (ValueError, ZeroDivisionError, IndexError) as e:
            print(
                f"⚠️  Error parsing score '{evaluation.overall_score}' - skipping this example: {e}"
            )
            numeric_score = 0.0

        return dspy.Prediction(
            score=numeric_score,
            feedback=evaluation.feedback_analysis,
        )

    except Exception as e:
        print(
            f"⚠️  Error in metric function - skipping this example: {type(e).__name__}: {str(e)[:100]}"
        )
        return dspy.Prediction(
            score=0.0, feedback="Evaluation error - example skipped"
        )


# ============================================================
# 5. RUN GEPA OPTIMIZATION OR LOAD SAVED PROMPTS
# ============================================================

"""
USAGE:
------
1. To run GEPA optimization (generate new prompts):
   - Set RUN_OPTIMIZATION = True
   - Optionally adjust categories_to_skip
   - Run the script

2. To evaluate using saved prompts (skip optimization):
   - Set RUN_OPTIMIZATION = False
   - Set prompt_file to your saved prompts file
   - Run the script - it will load prompts and run test evaluation only
"""

# Set this to True to run optimization, False to load saved prompts
RUN_OPTIMIZATION = False

# Thread-safe file I/O
file_write_lock = threading.Lock()

optimized_prompts = {}
optimized_students = {}
iteration_prompts = {}  # Track prompts at each iteration

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"auto_light_swebench_optimized_judge_prompts_{timestamp}.json"
iteration_file = f"auto_light_swebench_prompt_iterations_{timestamp}.json"


# ============================================================
# CATEGORY OPTIMIZATION FUNCTION (for parallel execution)
# ============================================================


def optimize_category(category, category_examples):
    """Optimize a single category with GEPA. Thread-safe."""
    try:
        thread_name = threading.current_thread().name
        log(f"\n{'=' * 60}")
        log(f"OPTIMIZING CATEGORY: {category} (Thread: {thread_name})")
        log(f"{'=' * 60}")

        # A. Get the starting prompt from your TruLens feedback system
        feedback_class_name = error_feedback_mapping[category]
        starting_instruction = getattr(
            feedback_v2, feedback_class_name
        ).system_prompt

        # B. Create a dspy.Predict instance
        student = dspy.Predict(StudentJudgeSignature)

        # C. Inject your existing TruLens prompt as the starting point
        student.signature = student.signature.with_instructions(
            starting_instruction
        )

        # D. Test BEFORE optimization
        log("\n--- Testing BEFORE optimization ---")
        test_ex = category_examples[0]
        log(f"test file: {test_ex.file}")
        log(f"category: {category}")
        log(
            f"Test trace length: {len(test_ex.trace)} chars (~{len(test_ex.trace) / 4:.0f} tokens)"
        )

        try:
            baseline_pred = student(trace=test_ex.trace)
        except Exception as e:
            log(f"ERROR during baseline prediction: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            log(
                "\n⚠️  Skipping baseline test and continuing with optimization..."
            )
            # Create a dummy prediction to continue
            baseline_pred = dspy.Prediction(
                critique="[Baseline prediction failed]"
            )
            baseline_result = dspy.Prediction(
                score=0.0, feedback="Baseline failed"
            )

        if "baseline prediction failed" not in baseline_pred.critique.lower():
            try:
                baseline_result = gepa_metric_with_feedback(
                    gold=test_ex, pred=baseline_pred
                )
                log(f"\nBaseline score: {baseline_result.score:.2f}")
                log(f"Baseline feedback: {baseline_result.feedback}...")
            except Exception as e:
                log(f"⚠️  Error evaluating baseline: {e}")
                baseline_result = dspy.Prediction(
                    score=0.0, feedback="Evaluation failed"
                )
        else:
            log("\nBaseline prediction failed, skipping evaluation")

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
            auto="light",  # For testing.
            # Tracking - track_stats keeps metrics in memory (no pickle if log_dir=None)
            track_stats=True,  # Track optimization statistics in memory
            # Parallelization - set to 1 to avoid rate limiting
            num_threads=2,
            # Disable warning about score mismatch (expected for LLM-as-judge metrics)
            warn_on_score_mismatch=False,
        )

        try:
            optimized_student = optimizer.compile(
                student=student, trainset=train_set, valset=val_set
            )
            print("✓ GEPA optimization completed successfully")

            # Save the optimized prompt info
            iteration_prompts[category] = {
                "starting_prompt": starting_instruction,
                "final_prompt": optimized_student.signature.instructions,
            }

            # Access tracked stats if available (kept in memory)
            if hasattr(optimizer, "stats") and optimizer.stats:
                log("\n  📊 Optimization Statistics:")
                stats = optimizer.stats
                iteration_prompts[category]["stats"] = {}

                # Display key metrics (exact attributes may vary by GEPA version)
                if hasattr(stats, "num_iterations"):
                    log(f"     - Iterations completed: {stats.num_iterations}")
                    iteration_prompts[category]["stats"]["num_iterations"] = (
                        stats.num_iterations
                    )
                if hasattr(stats, "total_metric_calls"):
                    log(
                        f"     - Total metric calls: {stats.total_metric_calls}"
                    )
                    iteration_prompts[category]["stats"][
                        "total_metric_calls"
                    ] = stats.total_metric_calls
                if hasattr(stats, "best_score"):
                    log(f"     - Best score achieved: {stats.best_score:.3f}")
                    iteration_prompts[category]["stats"]["best_score"] = float(
                        stats.best_score
                    )
                if hasattr(stats, "scores"):
                    log(f"     - Score progression: {stats.scores}")
                    iteration_prompts[category]["stats"]["scores"] = (
                        stats.scores
                    )

            # Save iteration tracking after each category
            with file_write_lock:
                with open(iteration_file, "w") as f:
                    json.dump(iteration_prompts, f, indent=2)
            log(f"  ✓ Saved iteration metadata to {iteration_file}")

        except Exception as e:
            log(f"⚠️  GEPA optimization failed: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            log(f"Category {category} failed, returning early...")
            return (category, None, None)

        # G. Test AFTER optimization on VALIDATION SET
        log("\n--- Testing AFTER optimization (on validation set) ---")
        log(
            f"\nEvaluating optimized judge on {len(val_set)} validation traces for {category}:"
        )

        all_scores_before = []
        all_scores_after = []

        for i, ex in enumerate(val_set):
            log(f"\nValidation example {i + 1}/{len(val_set)}: {ex['file']}")

            # Test unoptimized with error handling
            try:
                baseline_pred_ex = student(trace=ex.trace)
                result_before = gepa_metric_with_feedback(
                    gold=ex, pred=baseline_pred_ex
                )
                score_before = result_before.score
                log(f"  ✓ Baseline: {score_before:.2f}")
            except Exception as e:
                log(f"  ⚠️  Baseline failed: {type(e).__name__}: {str(e)[:100]}")
                log("  → Assigning score 0.0 and continuing to next example")
                score_before = 0.0

            all_scores_before.append(score_before)

            # Test optimized with error handling
            try:
                optimized_pred_ex = optimized_student(trace=ex.trace)
                result_after = gepa_metric_with_feedback(
                    gold=ex, pred=optimized_pred_ex
                )
                score_after = result_after.score
                log(f"  ✓ Optimized: {score_after:.2f}")
            except Exception as e:
                log(
                    f"  ⚠️  Optimized failed: {type(e).__name__}: {str(e)[:100]}"
                )
                log("  → Assigning score 0.0 and continuing to next example")
                score_after = 0.0

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

        # H. Save the optimized prompt (with thread-safe file I/O)
        final_prompt = optimized_student.signature.instructions

        # Use lock for thread-safe file writes
        with file_write_lock:
            optimized_prompts[category] = final_prompt
            optimized_students[category] = optimized_student

            # Save ALL optimized prompts so far (cumulative)
            with open(output_file, "w") as f:
                json.dump(optimized_prompts, f, indent=2)

            # Save iteration tracking
            with open(iteration_file, "w") as f:
                json.dump(iteration_prompts, f, indent=2)

        log(
            f"\n✓ Saved {len(optimized_prompts)} category prompt(s) to {output_file}"
        )

        log("\n--- Optimized prompt ---")
        log(final_prompt)

        return (category, optimized_student, final_prompt)

    except Exception as e:
        log(
            f"❌ Category {category} overall optimization failed: {type(e).__name__}: {e}"
        )
        import traceback

        traceback.print_exc()
        return (category, None, None)


if RUN_OPTIMIZATION:
    # ============================================================
    # RUN PARALLEL OPTIMIZATION WITH THREADPOOLEXECUTOR
    # ============================================================

    # Get categories to optimize (excluding skipped ones)
    categories_to_optimize = [
        (cat, examples)
        for cat, examples in grouped_examples.items()
        if cat not in ["PA", "PQ", "TS"]
    ]  # Adjust skip list as needed

    print(f"\n{'=' * 60}")
    print(
        f"Starting PARALLEL optimization of {len(categories_to_optimize)} categories"
    )
    print("Max parallel workers: 2")
    print("num_threads per GEPA: 2")
    print(f"{'=' * 60}\n")

    # Run optimizations in parallel with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(optimize_category, category, examples)
            for category, examples in categories_to_optimize
        ]

        # Wait for all to complete and collect results
        for future in futures:
            category, student, prompt = future.result()
            if student:
                print(f"✓ {category} optimization completed successfully")
            else:
                print(
                    f"✗ {category} optimization failed (continuing with other categories)"
                )

    print(f"\n{'=' * 60}")
    print("All parallel optimizations complete!")
    print(f"Successfully optimized: {list(optimized_students.keys())}")
    print(f"{'=' * 60}\n")

else:
    # ============================================================
    # LOAD SAVED PROMPTS AND CREATE STUDENTS
    # ============================================================

    log(f"\n{'=' * 60}")
    log("LOADING SAVED OPTIMIZED PROMPTS")
    log(f"{'=' * 60}\n")

    # Change this to the file containing your saved prompts
    prompt_file = "auto_light_swebench_optimized_judge_prompts_20251129_221543.json"  # Change to your saved prompts file

    try:
        with open(prompt_file, "r") as f:
            optimized_prompts = json.load(f)

        log(
            f"Loaded {len(optimized_prompts)} optimized prompts from {prompt_file}"
        )
        print(
            f"✓ Loaded {len(optimized_prompts)} optimized prompts from {prompt_file}"
        )

        # Create optimized students from saved prompts
        for category, optimized_prompt in optimized_prompts.items():
            log(f"\n{category}: Creating student with optimized prompt")
            log(f"  Prompt preview: {optimized_prompt[:200]}...")

            # Create a new student with the optimized prompt
            student = dspy.Predict(StudentJudgeSignature)
            student.signature = student.signature.with_instructions(
                optimized_prompt
            )
            optimized_students[category] = student

        log(f"\n{'=' * 60}")
        log(f"Created {len(optimized_students)} optimized students")
        log(f"{'=' * 60}\n")

        print(f"✓ Created {len(optimized_students)} optimized students")
        print(f"  Categories: {list(optimized_students.keys())}")

    except FileNotFoundError:
        log(f"❌ Error: Could not find {prompt_file}")
        log("Please run with RUN_OPTIMIZATION=True first to generate prompts.")
        print(f"❌ Error: Could not find {prompt_file}")
        print(
            "Please run with RUN_OPTIMIZATION=True first to generate prompts."
        )
        log_file.close()
        exit(1)

# ============================================================
# 6. SAVE FINAL OPTIMIZED PROMPTS (only if RUN_OPTIMIZATION=True)
# ============================================================

if RUN_OPTIMIZATION:
    print(f"\n{'=' * 60}")
    print("SAVING ALL OPTIMIZED PROMPTS")
    print(f"{'=' * 60}")

    # Save all final optimized prompts
    with open(output_file, "w") as f:
        json.dump(optimized_prompts, f, indent=2)
    print(
        f"✓ Saved {len(optimized_prompts)} optimized prompt(s) to: {output_file}"
    )

    # Save iteration metadata
    if iteration_prompts:
        with open(iteration_file, "w") as f:
            json.dump(iteration_prompts, f, indent=2)
        print(
            f"✓ Saved iteration metadata for {len(iteration_prompts)} category(s) to: {iteration_file}"
        )

    print(f"\nOptimized categories: {list(optimized_prompts.keys())}")

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

    total_caught_errors = 0
    total_errors = 0
    successful_predictions = 0
    failed_predictions = 0
    true_total_errors = 0

    for idx, ex in enumerate(test_examples):
        log(f"\nTest example {idx + 1}/{len(test_examples)}: {ex.file}")
        log(f"category: {category}")
        log(f"errors: {ex.errors}")
        log(f"num errors: {len(ex.errors)}")

        try:
            pred = optimized_student(trace=ex.trace)
            log(f"pred: {pred}")

            # Get the meta-judge evaluation
            golden_errors_str = "\n".join([f"- {err}" for err in ex.errors])
            evaluation = meta_judge(
                agent_trace=ex.trace,
                golden_errors=golden_errors_str,
                student_critique=pred.critique,
            )

            # Parse the caught/total from the score (e.g., "4/5")
            score_line = evaluation.overall_score.strip().split("\n")[0].strip()
            score_parts = score_line.split("/")
            caught_errors_in_example = int(score_parts[0])
            num_errors_in_example = int(score_parts[1])
            true_total_errors += len(ex.errors)
            # Add to totals
            total_caught_errors += caught_errors_in_example
            total_errors += num_errors_in_example
            successful_predictions += 1

            score_decimal = (
                caught_errors_in_example / num_errors_in_example
                if num_errors_in_example > 0
                else 0.0
            )
            log(
                f"  ✓ Test score: {score_decimal:.2f} ({caught_errors_in_example}/{num_errors_in_example} errors caught)"
            )
            log(f"  Feedback: {evaluation.feedback_analysis}")
        except Exception as e:
            log(
                f"  ⚠️  Test prediction failed: {type(e).__name__}: {str(e)[:100]}"
            )
            log("  → Skipping this example (not counting in final metrics)")
            log(f"ERROR: {e}")
            failed_predictions += 1

        log("********************************************************")
        time.sleep(1)  # Rate limiting: wait 1 second between test examples

    # Calculate and log final metrics
    if total_errors > 0:
        recall = total_caught_errors / total_errors
        log(f"\n{category}: Test Set Results:")
        log(f"  Total caught errors: {total_caught_errors}")
        log(f"  Total errors: {total_errors}")
        log(f"  True total errors: {true_total_errors}")
        log(f"  Recall: {recall:.3f} ({total_caught_errors}/{total_errors})")
        log(f"  Successful predictions: {successful_predictions}")
        log(f"  Failed predictions: {failed_predictions}")
    else:
        log(f"\n{category}: No successful predictions to evaluate")


# Close the log file
log_file.close()
