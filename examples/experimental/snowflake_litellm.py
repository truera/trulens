from collections import defaultdict
import os

import dspy
import pandas as pd
from trulens.feedback.v2 import feedback as feedback_v2

## set ENV variables
os.environ["SNOWFLAKE_ACCOUNT_ID"] = "SFCOGSOPS-SNOWHOUSE-AWS-US-WEST-2"
# os.environ["SNOWFLAKE_USER"] = "AJIA"
os.environ["SNOWFLAKE_JWT"] = "pat/"

# response = completion(
#     model="snowflake/claude-sonnet-4-5",
#     messages = [{ "content": "Hello, how are you?","role": "user"}],
# )
# print(response)


class StudentJudgeSignature(dspy.Signature):
    """Analyze an agent trace for errors based on a specific error category."""

    trace: str = dspy.InputField(desc="The raw execution trace of the agent.")

    critique: str = dspy.OutputField(
        desc="A detailed critique listing all errors found in the trace given the criteria for this category."
    )


# Ensure LM works properly
lm = dspy.LM(
    "snowflake/claude-sonnet-4-5",
    temperature=1,
    max_tokens=32000,
    api_key=os.environ["SNOWFLAKE_JWT"],
)
dspy.configure(lm=lm)

# Test the LM connection
print("Testing LM connection...")
try:
    response = lm("Hello, how are you?")
    print(f"✓ LM connection successful: {response[:100]}...")
except Exception as e:
    print(f"✗ LM connection FAILED: {type(e).__name__}: {e}")
    print("Please check your JWT token and network connection.")
    import sys

    sys.exit(1)

# For each metric, build grouped structure of errors per trace example
grouped_examples = defaultdict(list)

print("\nLoading data files...")
try:
    df = pd.read_csv(
        "examples/experimental/GPA Judge Error Analysis - TRAIN_CSV.csv"
    )
    print(f"✓ Loaded training CSV: {len(df)} rows")
except Exception as e:
    print(f"✗ Error loading training CSV: {e}")
    import sys

    sys.exit(1)

try:
    judge_df = pd.read_csv(
        "examples/experimental/TRAIL_GAIA_Judge_Output_Per_Trace.csv"
    )
    print(f"✓ Loaded judge CSV: {len(judge_df)} rows")
except Exception as e:
    print(f"✗ Error loading judge CSV: {e}")
    import sys

    sys.exit(1)


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
        try:
            trace_content = open(
                f"examples/experimental/{file_value}.txt",
                "r",
            ).read()

            example = dspy.Example(
                trace=trace_content,
                error_category=error_code,
                errors=errors_list,
                file=file_value,  # Keep for debugging
            ).with_inputs("trace")

            grouped_examples[error_code].append(example)
        except FileNotFoundError:
            print(
                f"⚠️  Warning: File not found: examples/experimental/{file_value}.txt"
            )
            continue
        except Exception as e:
            print(f"⚠️  Error loading {file_value}: {e}")
            continue

error_feedback_mapping = {
    "LC": "LogicalConsistency",
    "EE": "ExecutionEfficiency",
    "PA": "PlanAdherence",
    "PQ": "PlanQuality",
    "TC": "ToolCalling",
    "TS": "ToolSelection",
}

# For each metric, test the baseline prediction
for category, category_examples in grouped_examples.items():
    print(f"\n{'=' * 60}")
    print(f"Testing category: {category}")
    print(f"{'=' * 60}")
    if category != "PA":
        continue
    try:
        # A. Get the starting prompt from your TruLens feedback system
        feedback_class_name = error_feedback_mapping[category]
        starting_instruction = getattr(
            feedback_v2, feedback_class_name
        ).system_prompt

        test_ex = category_examples[0]
        print(f"filename: {test_ex.file}")
        print(
            f"Test trace length: {len(test_ex.trace) + len(starting_instruction)} chars (~{(len(test_ex.trace) + len(starting_instruction)) / 4:.0f} tokens)"
        )

        student = dspy.Predict(StudentJudgeSignature)
        student.signature = StudentJudgeSignature.with_instructions(
            starting_instruction
        )

        print(f"Running baseline prediction for {category}...")
        print(f"student signature: {student.signature}")
        print(f"trace: {test_ex.trace}")
        baseline_pred = student(trace=test_ex.trace)
        # Check if prediction has expected fields
        if hasattr(baseline_pred, "critique"):
            print("✓ Baseline prediction successful")
            print(f"Critique preview: {baseline_pred.critique[:200]}...")
        else:
            print("⚠️  Warning: Prediction missing 'critique' field")
            print(f"Prediction: {baseline_pred}")

    except Exception as e:
        print(f"⚠️  Error on category {category}: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        print(f"Skipping {category} and continuing to next category...")
        continue

    print("********************************************************")
