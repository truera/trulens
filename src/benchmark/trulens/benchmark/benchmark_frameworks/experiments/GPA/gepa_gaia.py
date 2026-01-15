"""
GEPA Judge Optimization and Evaluation Script.

USAGE:
------
1. To run GEPA optimization (generate new prompts):
   - Set RUN_OPTIMIZATION = True
   - Set OPTIMIZATION_CATEGORIES = ["PA", "LC"] for specific categories, or [] for all
   - Run the script

2. To evaluate using saved prompts (skip optimization):
   - Set RUN_OPTIMIZATION = False
   - Set INPUT_PROMPT_FILE to your saved prompts file
   - Run the script

Environment Variables Required:
    SNOWFLAKE_ACCOUNT: Snowflake account identifier
    SNOWFLAKE_JWT: JWT token for authentication
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import wraps
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

import dspy
from dspy.teleprompt import GEPA
import pandas as pd
from trulens.feedback.v2 import feedback as feedback_v2

# ============================================================
# CONFIGURATION
# ============================================================

# Snowflake credentials (from environment variables)
SNOWFLAKE_ACCOUNT = os.environ.get(
    "SNOWFLAKE_ACCOUNT", "SFCOGSOPS-SNOWHOUSE-AWS-US-WEST-2"
)
JWT_TOKEN = os.environ.get("SNOWFLAKE_JWT", "")

# LM Configuration
STUDENT_LM_MODEL = "openai/claude-sonnet-4-5"
META_LM_MODEL = "openai/claude-sonnet-4-5"
REFLECTION_LM_MODEL = "openai/claude-sonnet-4-5"
LM_TEMPERATURE_DEFAULT = 1.0
LM_TEMPERATURE_EVAL = 0.0
LM_MAX_TOKENS = 32000

# Paths
DATA_DIR = Path(
    "src/benchmark/trulens/benchmark/benchmark_frameworks/experiments/GPA"
)
TRAIN_CSV = DATA_DIR / "GPA Judge Error Analysis - TRAIN_CSV.csv"
TEST_CSV = DATA_DIR / "GPA Judge Error Analysis - TEST_CSV.csv"
JUDGE_OUTPUT_CSV = DATA_DIR / "TRAIL_GAIA_Judge_Output_Per_Trace.csv"
VALIDATION_RESULTS_FILE = DATA_DIR / "lax_validation_results.txt"
TRACE_DIR = DATA_DIR / "GAIA"

# Optimization settings
RUN_OPTIMIZATION = True
OPTIMIZATION_CATEGORIES: list[str] = [
    "PA"
]  # Empty list = all categories, non-empty = only these
MAX_PARALLEL_WORKERS = 2
GEPA_NUM_THREADS = 2
INPUT_PROMPT_FILE = "auto-medium_gaia_prompts.json"

# Retry settings
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2

# Output file names (timestamp will be appended for parallel optimization)
LOG_FILE_PREFIX = "auto-medium-fixed-metajudge-gaia"
OUTPUT_PROMPT_FILE = "gaia_optimized_judge_prompts"
ITERATION_FILE_PREFIX = "gaia_prompt_iterations"

# Debug settings
TEST_SINGLE_EXAMPLE = False
SINGLE_EXAMPLE_CATEGORY = "PA"
SINGLE_EXAMPLE_FILE = "18efa24e637b9423f34180d1f2041d3e"  # Just the filename (no directory prefix)

# Error category mappings
ERROR_FEEDBACK_MAPPING = {
    "LC": "LogicalConsistency",
    "EE": "ExecutionEfficiency",
    "PA": "PlanAdherence",
    "PQ": "PlanQuality",
    "TC": "ToolCalling",
    "TS": "ToolSelection",
}


# ============================================================
# RETRY LOGIC
# ============================================================


def retry_with_backoff(
    max_retries: int = MAX_RETRIES, initial_delay: int = INITIAL_RETRY_DELAY
) -> Callable:
    """Decorator to retry API calls with exponential backoff on 429 errors."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "rate limit" in error_str.lower():
                        if attempt < max_retries - 1:
                            delay = initial_delay * (2**attempt)
                            print(
                                f"⚠️  Rate limit hit (attempt {attempt + 1}/{max_retries}), "
                                f"waiting {delay}s..."
                            )
                            time.sleep(delay)
                            continue
                    raise
            raise Exception(f"Failed after {max_retries} retries")

        return wrapper

    return decorator


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def get_api_base() -> str:
    """Get the Snowflake API base URL."""
    return (
        f"https://{SNOWFLAKE_ACCOUNT}.snowflakecomputing.com/api/v2/cortex/v1"
    )


def create_lm(
    model: str = STUDENT_LM_MODEL, temperature: float = LM_TEMPERATURE_DEFAULT
) -> dspy.LM:
    """Create a configured language model instance with retry logic."""
    lm = dspy.LM(
        model=model,
        temperature=temperature,
        max_completion_tokens=LM_MAX_TOKENS,
        api_key=JWT_TOKEN,
        api_base=get_api_base(),
    )
    # Wrap LM calls with retry logic for rate limit handling
    original_call = lm.__call__
    lm.__call__ = retry_with_backoff()(original_call)
    return lm


def read_trace_file(filename: str) -> str:
    """Read a trace file from the trace directory."""
    filepath = TRACE_DIR / f"{filename}.txt"
    with open(filepath) as f:
        return f.read()


def parse_gpa_category(value: str | float | None) -> list[str]:
    """
    Parse GPA Category column to list of error codes.

    Handles comma-separated values like "LC, EE" or "LC,EE".
    Returns empty list for null/empty values.
    """
    if (
        pd.isna(value)
        or value == ""
        or (isinstance(value, str) and not value.strip())
    ):
        return []

    if isinstance(value, str):
        codes = [code.strip() for code in value.split(",")]
        return [code for code in codes if code]

    return []


def parse_score_string(score_str: str) -> tuple[int, int]:
    """
    Parse a score string like "3/5" into (caught, total) tuple.

    Takes the first line and parses the fraction.
    """
    score_line = score_str.strip().split("\n")[0].strip()
    parts = score_line.split("/")
    return int(parts[0]), int(parts[1])


def format_golden_errors(errors: list[str]) -> str:
    """Format a list of errors as a bullet-point string."""
    return "\n".join(f"- {err}" for err in errors)


# ============================================================
# LOGGING
# ============================================================


class Logger:
    """Thread-safe file logger with optional console output."""

    def __init__(self, log_path: Path, console_output: bool = False):
        import threading

        self.log_file = open(log_path, "w", encoding="utf-8")
        self.console_output = console_output
        self._lock = threading.Lock()

    def log(self, message: str = "") -> None:
        """Write message to log file in a thread-safe manner."""
        with self._lock:
            print(message, file=self.log_file, flush=True)
            if self.console_output:
                print(message)

    def close(self) -> None:
        """Close the log file."""
        self.log_file.close()

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ============================================================
# DATA LOADING
# ============================================================


def strip_trace_prefix(filename: str) -> str:
    """Strip the 'GAIA/' prefix from filenames if present."""
    prefix = "GAIA/"
    if filename.startswith(prefix):
        return filename[len(prefix) :]
    return filename


def load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and prepare the train, test, and judge output dataframes."""
    df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    judge_df = pd.read_csv(JUDGE_OUTPUT_CSV)

    # Strip GAIA/ prefix from filenames and create a clean 'file' column
    df["file"] = df["Filename"].apply(strip_trace_prefix)
    test_df["file"] = test_df["Filename"].apply(strip_trace_prefix)

    # Load trace content for train data
    for i in range(len(df)):
        df.loc[i, "trace"] = read_trace_file(df.iloc[i]["file"])

    # Load trace content for test data
    for i in range(len(test_df)):
        test_df.loc[i, "trace"] = read_trace_file(test_df.iloc[i]["file"])

    # Parse GPA categories
    for dataframe in [df, test_df]:
        dataframe["gpa_list"] = dataframe["Align_Judges"].apply(
            parse_gpa_category
        )
        dataframe["caught_list"] = dataframe["Caught"].apply(parse_gpa_category)
        dataframe["aligned_list"] = dataframe["Align_Judges"].apply(
            parse_gpa_category
        )

    return df, test_df, judge_df


def build_grouped_examples(
    df: pd.DataFrame, data_dir: Path = DATA_DIR
) -> dict[str, list[dspy.Example]]:
    """
    Build examples grouped by error category from a dataframe.

    Each example contains the trace, error category, list of errors, and filename.
    """
    grouped = defaultdict(list)

    for file_value, file_group in df.groupby("file"):
        # Get all unique error codes in this trace
        all_error_codes: set[str] = set()
        for gpa_list in file_group["gpa_list"]:
            all_error_codes.update(gpa_list)

        # For each error code, create a dspy.Example
        for error_code in all_error_codes:
            rows_with_error = file_group[
                file_group["gpa_list"].apply(lambda x, ec=error_code: ec in x)
            ]

            # Collect all errors from these rows
            errors_list = []
            for idx in rows_with_error.index:
                raw_error = (
                    df.loc[idx, "Raw Error"]
                    if "Raw Error" in df.columns
                    else None
                )
                if pd.notna(raw_error) and str(raw_error).strip():
                    errors_list.append(raw_error)

            example = dspy.Example(
                trace=read_trace_file(file_value),
                error_category=error_code,
                errors=errors_list,
                file=file_value,
            ).with_inputs("trace")

            grouped[error_code].append(example)

    return dict(grouped)


# ============================================================
# DSPY SIGNATURES
# ============================================================


class StudentJudgeSignature(dspy.Signature):
    """Analyze an agent trace for errors based on a specific error category."""

    trace: str = dspy.InputField(desc="The raw execution trace of the agent.")
    critique: str = dspy.OutputField(
        desc="A detailed critique listing all errors found in the trace "
        "given the criteria for this category."
    )


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


# ============================================================
# METRIC AND EVALUATION
# ============================================================

# Module-level meta_judge instance (initialized after dspy.configure())
_meta_judge: dspy.ChainOfThought | None = None


def get_meta_judge() -> dspy.ChainOfThought:
    """Get the module-level meta_judge instance."""
    global _meta_judge
    if _meta_judge is None:
        _meta_judge = dspy.ChainOfThought(MetaJudgeSignature)
    return _meta_judge


def gepa_metric_with_feedback(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
    category: str | None = None,
) -> dspy.Prediction:
    """
    Metric for GEPA optimization.

    GEPA requires this signature: (gold, pred, trace, pred_name, pred_trace)

    Args:
        gold: dspy.Example with .trace, .error_category, .errors (ground truth)
        pred: The student judge's output with .critique
        trace: DSPy's internal execution trace (unused)
        pred_name: Name of the prediction field (unused)
        pred_trace: Trace of the prediction (unused)
        category: Error category (unused)

    Returns:
        dspy.Prediction with 'score' and 'feedback' fields
    """
    # Check if prediction has the required field
    if not hasattr(pred, "critique"):
        print("⚠️  Prediction missing 'critique' field - skipping this example")
        print(f"   Prediction fields: {list(pred.__dict__.keys())}")
        return dspy.Prediction(
            score=0.0, feedback="Missing critique field - example skipped"
        )

    meta_judge = get_meta_judge()
    golden_errors_str = format_golden_errors(gold.errors)

    try:
        with dspy.settings.context(
            lm=create_lm(model=META_LM_MODEL, temperature=LM_TEMPERATURE_EVAL)
        ):
            evaluation = meta_judge(
                agent_trace=gold.trace,
                golden_errors=golden_errors_str,
                student_critique=pred.critique,
            )
    except Exception as e:
        print(
            f"⚠️  Meta-judge failed - skipping this example: "
            f"{type(e).__name__}: {str(e)[:100]}"
        )
        return dspy.Prediction(
            score=0.0, feedback="Meta-judge error - example skipped"
        )

    try:
        caught, total = parse_score_string(evaluation.overall_score)
        numeric_score = caught / total if total > 0 else 0.0
    except (ValueError, ZeroDivisionError, IndexError) as e:
        print(
            f"⚠️  Error parsing score '{evaluation.overall_score}' - "
            f"skipping this example: {e}"
        )
        numeric_score = 0.0

    return dspy.Prediction(
        score=numeric_score,
        feedback=evaluation.feedback_analysis,
    )


def evaluate_prediction(
    example: dspy.Example,
    student: dspy.Predict,
    meta_judge: dspy.ChainOfThought,
    logger: Logger,
) -> tuple[int, int, bool]:
    """
    Evaluate a single prediction and return caught/total errors.

    Returns:
        Tuple of (caught_errors, total_errors, success)
    """
    logger.log(f"errors: {example.errors}")
    logger.log(f"num errors: {len(example.errors)}")

    try:
        pred = student(trace=example.trace)
        logger.log(f"pred: {pred}")

        golden_errors_str = format_golden_errors(example.errors)

        with dspy.settings.context(
            lm=create_lm(model=META_LM_MODEL, temperature=LM_TEMPERATURE_EVAL)
        ):
            evaluation = meta_judge(
                agent_trace=example.trace,
                golden_errors=golden_errors_str,
                student_critique=pred.critique,
            )

        caught, total = parse_score_string(evaluation.overall_score)
        score_decimal = caught / total if total > 0 else 0.0

        logger.log(
            f"  ✓ Test score: {score_decimal:.2f} ({caught}/{total} errors caught)"
        )
        logger.log(f"  Feedback: {evaluation.feedback_analysis}")

        return caught, total, True

    except Exception as e:
        logger.log(
            f"  ⚠️  Test prediction failed: {type(e).__name__}: {str(e)[:100]}"
        )
        logger.log("  → Skipping this example (not counting in final metrics)")
        logger.log(f"ERROR: {e}")
        return 0, 0, False


# ============================================================
# OPTIMIZATION
# ============================================================


def optimize_category(
    category: str,
    examples: list[dspy.Example],
    logger: Logger,
    output_file: Path,
    iteration_file: Path,
    optimized_prompts: dict[str, str],
    optimized_students: dict[str, dspy.Predict],
    iteration_prompts: dict[str, dict],
) -> tuple[str, dspy.Predict | None, str | None]:
    """Optimize a single category with GEPA. Thread-safe."""
    import threading

    try:
        thread_name = threading.current_thread().name
        logger.log(f"\n{'=' * 60}")
        logger.log(f"OPTIMIZING CATEGORY: {category} (Thread: {thread_name})")
        logger.log(f"{'=' * 60}")

        # Get the starting prompt from TruLens feedback system
        feedback_class_name = ERROR_FEEDBACK_MAPPING[category]
        starting_instruction = getattr(
            feedback_v2, feedback_class_name
        ).system_prompt
        logger.log(f"{starting_instruction[:200]}...")

        # Create and configure student
        student = dspy.Predict(StudentJudgeSignature)
        student.signature = student.signature.with_instructions(
            starting_instruction
        )

        # Test BEFORE optimization
        logger.log("\n--- Testing BEFORE optimization ---")
        test_ex = examples[0]
        logger.log(f"test file: {test_ex.file}")
        logger.log(f"category: {category}")
        logger.log(
            f"Test trace length: {len(test_ex.trace)} chars "
            f"(~{len(test_ex.trace) / 4:.0f} tokens)"
        )

        try:
            baseline_pred = student(trace=test_ex.trace)
            baseline_result = gepa_metric_with_feedback(
                gold=test_ex, pred=baseline_pred
            )
            logger.log(f"\nBaseline score: {baseline_result.score:.2f}")
            logger.log(f"Baseline feedback: {baseline_result.feedback}...")
        except Exception as e:
            logger.log(
                f"ERROR during baseline prediction: {type(e).__name__}: {e}"
            )
            logger.log(
                "\n⚠️  Skipping baseline test and continuing with optimization..."
            )

        # Split into train/val
        if len(examples) >= 10:
            split_idx = int(len(examples) * 0.8)
            train_set = examples[:split_idx]
            val_set = examples[split_idx:]
        else:
            logger.log(
                f"  ⚠️  Small dataset ({len(examples)} examples) - "
                "using all for train & val"
            )
            train_set = examples
            val_set = examples

        logger.log(
            f"\nTrain: {len(train_set)} examples, Val: {len(val_set)} examples"
        )

        # Run GEPA optimization
        logger.log("\n--- Running GEPA optimization ---")
        optimizer = GEPA(
            metric=gepa_metric_with_feedback,
            reflection_lm=create_lm(
                model=REFLECTION_LM_MODEL, temperature=LM_TEMPERATURE_DEFAULT
            ),
            auto="light",
            num_threads=GEPA_NUM_THREADS,
        )

        try:
            optimized_student = optimizer.compile(
                student=student, trainset=train_set, valset=val_set
            )
            print(f"✓ GEPA optimization for {category} completed successfully")

            # Save iteration metadata
            iteration_prompts[category] = {
                "starting_prompt": starting_instruction,
                "final_prompt": optimized_student.signature.instructions,
            }

            # Save iteration tracking
            with open(iteration_file, "w") as f:
                json.dump(iteration_prompts, f, indent=2)
            logger.log(f"  ✓ Saved iteration metadata to {iteration_file}")

        except Exception as e:
            logger.log(f"⚠️  GEPA optimization failed: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            logger.log(f"Category {category} failed, returning early...")
            return (category, None, None)

        # Test AFTER optimization on validation set
        logger.log("\n--- Testing AFTER optimization (on validation set) ---")
        logger.log(
            f"\nEvaluating optimized judge on {len(val_set)} validation traces "
            f"for {category}:"
        )

        all_scores_before = []
        all_scores_after = []

        for i, ex in enumerate(val_set):
            logger.log(
                f"\nValidation example {i + 1}/{len(val_set)}: {ex['file']}"
            )

            # Test unoptimized
            try:
                baseline_pred_ex = student(trace=ex.trace)
                result_before = gepa_metric_with_feedback(
                    gold=ex, pred=baseline_pred_ex
                )
                score_before = result_before.score
                logger.log(f"  ✓ Baseline: {score_before:.2f}")
            except Exception as e:
                logger.log(
                    f"  ⚠️  Baseline failed: {type(e).__name__}: {str(e)[:100]}"
                )
                logger.log(
                    "  → Assigning score 0.0 and continuing to next example"
                )
                score_before = 0.0

            all_scores_before.append(score_before)

            # Test optimized
            try:
                optimized_pred_ex = optimized_student(trace=ex.trace)
                result_after = gepa_metric_with_feedback(
                    gold=ex, pred=optimized_pred_ex
                )
                score_after = result_after.score
                logger.log(f"  ✓ Optimized: {score_after:.2f}")
            except Exception as e:
                logger.log(
                    f"  ⚠️  Optimized failed: {type(e).__name__}: {str(e)[:100]}"
                )
                logger.log(
                    "  → Assigning score 0.0 and continuing to next example"
                )
                score_after = 0.0

            all_scores_after.append(score_after)

            logger.log(
                f"  Trace {i + 1} ({ex['file']}): {score_before:.2f} → "
                f"{score_after:.2f} ({score_after - score_before:+.2f})"
            )
            time.sleep(1)  # Rate limiting

        avg_before = sum(all_scores_before) / len(all_scores_before)
        avg_after = sum(all_scores_after) / len(all_scores_after)

        logger.log(
            f"\n  📊 Average Score: {avg_before:.2f} → {avg_after:.2f} "
            f"({avg_after - avg_before:+.2f})"
        )

        # Save the optimized prompt
        final_prompt = optimized_student.signature.instructions
        optimized_prompts[category] = final_prompt
        optimized_students[category] = optimized_student

        # Save all prompts (cumulative)
        with open(output_file, "w") as f:
            json.dump(optimized_prompts, f, indent=2)

        with open(iteration_file, "w") as f:
            json.dump(iteration_prompts, f, indent=2)

        logger.log(
            f"\n✓ Saved {len(optimized_prompts)} category prompt(s) to {output_file}"
        )
        logger.log("\n--- Optimized prompt ---")
        logger.log(final_prompt)

        return (category, optimized_student, final_prompt)

    except Exception as e:
        logger.log(
            f"❌ Category {category} overall optimization failed: "
            f"{type(e).__name__}: {e}"
        )
        import traceback

        traceback.print_exc()
        return (category, None, None)


def run_parallel_optimization(
    grouped_examples: dict[str, list[dspy.Example]],
    logger: Logger,
    output_file: Path,
    iteration_file: Path,
) -> tuple[dict[str, str], dict[str, dspy.Predict]]:
    """Run GEPA optimization in parallel for multiple categories."""
    optimized_prompts: dict[str, str] = {}
    optimized_students: dict[str, dspy.Predict] = {}
    iteration_prompts: dict[str, dict] = {}

    # Get categories to optimize
    if OPTIMIZATION_CATEGORIES:
        # Optimize only specified categories
        categories_to_optimize = [
            (cat, examples)
            for cat, examples in grouped_examples.items()
            if cat in OPTIMIZATION_CATEGORIES
        ]
    else:
        # Optimize all categories
        categories_to_optimize = list(grouped_examples.items())

    print(f"\n{'=' * 60}")
    print(
        f"Starting PARALLEL optimization of {len(categories_to_optimize)} categories"
    )
    print(f"Max parallel workers: {MAX_PARALLEL_WORKERS}")
    print(f"num_threads per GEPA: {GEPA_NUM_THREADS}")
    print(f"{'=' * 60}\n")

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        futures = [
            executor.submit(
                optimize_category,
                category,
                examples,
                logger,
                output_file,
                iteration_file,
                optimized_prompts,
                optimized_students,
                iteration_prompts,
            )
            for category, examples in categories_to_optimize
        ]

        for future in futures:
            category, student, _ = future.result()
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

    return optimized_prompts, optimized_students


def load_optimized_prompts(
    prompt_file: str, logger: Logger
) -> dict[str, dspy.Predict]:
    """
    Load saved prompts and create student instances.

    Returns:
        Dictionary mapping category to optimized student
    """
    logger.log(f"\n{'=' * 60}")
    logger.log("LOADING SAVED OPTIMIZED PROMPTS")
    logger.log(f"{'=' * 60}\n")

    with open(prompt_file) as f:
        optimized_prompts = json.load(f)

    logger.log(
        f"Loaded {len(optimized_prompts)} optimized prompts from {prompt_file}"
    )
    print(
        f"✓ Loaded {len(optimized_prompts)} optimized prompts from {prompt_file}"
    )

    optimized_students: dict[str, dspy.Predict] = {}
    for category, prompt in optimized_prompts.items():
        logger.log(f"\n{category}: Creating student with optimized prompt")
        logger.log(f"  Prompt preview: {prompt[:200]}...")

        student = dspy.Predict(StudentJudgeSignature)
        student.signature = student.signature.with_instructions(prompt)
        optimized_students[category] = student

    logger.log(f"\n{'=' * 60}")
    logger.log(f"Created {len(optimized_students)} optimized students")
    logger.log(f"{'=' * 60}\n")

    print(f"✓ Created {len(optimized_students)} optimized students")
    print(f"  Categories: {list(optimized_students.keys())}")

    return optimized_students


# ============================================================
# TEST EVALUATION
# ============================================================


def run_test_evaluation(
    optimized_students: dict[str, dspy.Predict],
    test_grouped_examples: dict[str, list[dspy.Example]],
    logger: Logger,
) -> None:
    """Run evaluation on the test set for all categories."""
    logger.log(f"\n{'=' * 60}")
    logger.log("FINAL EVALUATION ON TEST SET")
    logger.log(f"{'=' * 60}\n")

    meta_judge = get_meta_judge()

    for category, student in optimized_students.items():
        test_examples = test_grouped_examples.get(category, [])
        if not test_examples:
            logger.log(f"{category}: No test examples")
            continue

        total_caught = 0
        total_errors = 0
        true_total_errors = 0
        successful = 0
        failed = 0

        for idx, ex in enumerate(test_examples):
            logger.log(
                f"\nTest example {idx + 1}/{len(test_examples)}: {ex.file}"
            )
            logger.log(f"category: {category}")

            caught, total, success = evaluate_prediction(
                ex, student, meta_judge, logger
            )

            if success:
                total_caught += caught
                total_errors += total
                true_total_errors += len(ex.errors)
                successful += 1
            else:
                failed += 1

            logger.log("*" * 56)
            time.sleep(1)  # Rate limiting

        # Log final metrics
        if total_errors > 0:
            recall = total_caught / total_errors
            logger.log(f"\n{category}: Test Set Results:")
            logger.log(f"  Total caught errors: {total_caught}")
            logger.log(f"  Total errors: {total_errors}")
            logger.log(f"  True total errors: {true_total_errors}")
            logger.log(
                f"  Recall: {recall:.3f} ({total_caught}/{total_errors})"
            )
            logger.log(f"  Successful predictions: {successful}")
            logger.log(f"  Failed predictions: {failed}")
        else:
            logger.log(f"\n{category}: No successful predictions to evaluate")


def test_single_example(
    optimized_students: dict[str, dspy.Predict],
    test_grouped_examples: dict[str, list[dspy.Example]],
    category: str,
    target_file: str,
    logger: Logger,
) -> None:
    """Test a single specific example for debugging."""
    logger.log(f"\n{'=' * 60}")
    logger.log(f"TESTING SINGLE EXAMPLE: {target_file}")
    logger.log(f"{'=' * 60}\n")

    if category not in optimized_students:
        logger.log(f"Category {category} not in optimized_students")
        logger.log(f"Available categories: {list(optimized_students.keys())}")
        return

    student = optimized_students[category]
    test_examples = test_grouped_examples.get(category, [])

    target_example = next(
        (ex for ex in test_examples if ex.file == target_file), None
    )

    if not target_example:
        logger.log(f"Could not find example with file: {target_file}")
        logger.log(
            f"Available {category} examples: {[ex.file for ex in test_examples]}"
        )
        return

    logger.log(f"Found example: {target_example.file}")
    logger.log(f"Category: {category}")

    meta_judge = get_meta_judge()

    caught, total, success = evaluate_prediction(
        target_example, student, meta_judge, logger
    )

    if success:
        logger.log("\n✓ Single example evaluation complete")
        logger.log(f"  Result: {caught}/{total} errors caught")
    else:
        logger.log("\n✗ Single example evaluation failed")


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Main entry point for the script."""
    # Validate JWT token
    if not JWT_TOKEN:
        print("Error: SNOWFLAKE_JWT environment variable not set")
        sys.exit(1)

    # Set environment variables for Snowflake
    os.environ["SNOWFLAKE_ACCOUNT"] = SNOWFLAKE_ACCOUNT

    # Configure DSPy
    lm = create_lm(model=STUDENT_LM_MODEL, temperature=LM_TEMPERATURE_DEFAULT)
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    dspy.configure(lm=lm)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"{LOG_FILE_PREFIX}_{timestamp}.log")
    output_file = Path(f"{OUTPUT_PROMPT_FILE}_{timestamp}.json")
    iteration_file = Path(f"{ITERATION_FILE_PREFIX}_{timestamp}.json")

    with Logger(log_path, console_output=False) as logger:
        # Load data
        train_df, test_df, _judge_df = load_dataframes()
        grouped_examples = build_grouped_examples(train_df)
        test_grouped_examples = build_grouped_examples(test_df)

        # Run optimization or load saved prompts
        if RUN_OPTIMIZATION:
            optimized_prompts, optimized_students = run_parallel_optimization(
                grouped_examples, logger, output_file, iteration_file
            )

            # Save final prompts
            print(f"\n{'=' * 60}")
            print("SAVING ALL OPTIMIZED PROMPTS")
            print(f"{'=' * 60}")

            with open(output_file, "w") as f:
                json.dump(optimized_prompts, f, indent=2)
            print(
                f"✓ Saved {len(optimized_prompts)} optimized prompt(s) to: {output_file}"
            )
            print(f"\nOptimized categories: {list(optimized_prompts.keys())}")
        else:
            try:
                optimized_students = load_optimized_prompts(
                    INPUT_PROMPT_FILE, logger
                )
            except FileNotFoundError:
                logger.log(f"❌ Error: Could not find {INPUT_PROMPT_FILE}")
                logger.log(
                    "Please run with RUN_OPTIMIZATION=True first to generate prompts."
                )
                print(f"❌ Error: Could not find {INPUT_PROMPT_FILE}")
                print(
                    "Please run with RUN_OPTIMIZATION=True first to generate prompts."
                )
                sys.exit(1)

        # Run single example test or full evaluation
        if TEST_SINGLE_EXAMPLE:
            test_single_example(
                optimized_students,
                test_grouped_examples,
                SINGLE_EXAMPLE_CATEGORY,
                SINGLE_EXAMPLE_FILE,
                logger,
            )
        else:
            run_test_evaluation(
                optimized_students, test_grouped_examples, logger
            )


if __name__ == "__main__":
    main()
