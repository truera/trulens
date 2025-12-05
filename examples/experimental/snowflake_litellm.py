from collections import defaultdict
import os

import dspy
import pandas as pd
from trulens.feedback.v2 import feedback as feedback_v2

## set ENV variables
os.environ["SNOWFLAKE_ACCOUNT_ID"] = "SFCOGSOPS-SNOWHOUSE-AWS-US-WEST-2"
# os.environ["SNOWFLAKE_USER"] = "AJIA"
os.environ["SNOWFLAKE_JWT"] = "pat/..."

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


lm = dspy.LM(
    "snowflake/claude-sonnet-4-5",
    temperature=1,
    max_tokens=32000,
    api_key=os.environ["SNOWFLAKE_JWT"],
)
dspy.configure(lm=lm)

response = lm("Hello, how are you?")

print(response)

# Build grouped structure directly
grouped_examples = defaultdict(list)

df = pd.read_csv(
    "examples/experimental/GPA Judge Error Analysis - TRAIN_CSV.csv"
)
judge_df = pd.read_csv(
    "examples/experimental/TRAIL_GAIA_Judge_Output_Per_Trace.csv"
)


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

error_feedback_mapping = {
    "LC": "LogicalConsistency",
    "EE": "ExecutionEfficiency",
    "PA": "PlanAdherence",
    "PQ": "PlanQuality",
    "TC": "ToolCalling",
    "TS": "ToolSelection",
}

for category, category_examples in grouped_examples.items():
    # A. Get the starting prompt from your TruLens feedback system
    feedback_class_name = error_feedback_mapping[category]
    starting_instruction = getattr(
        feedback_v2, feedback_class_name
    ).system_prompt
    test_ex = category_examples[0]
    student = dspy.Predict(StudentJudgeSignature)
    student.signature = StudentJudgeSignature.with_instructions(
        starting_instruction
    )
    baseline_pred = student(trace=test_ex.trace)
    print(f"baseline prediction for {category}")
    print(baseline_pred)
    print("********************************************************")
