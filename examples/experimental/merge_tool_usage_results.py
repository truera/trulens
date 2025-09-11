#!/usr/bin/env python3
"""
Script to merge Tool Judge results with existing benchmark data.

This script reads the existing benchmark CSV and the new tool judge results CSV,
then creates a new merged CSV file with Tool Selection_score and Tool Selection_reasons
and Tool Calling_score and Tool Calling_reasons
columns appended to the existing data.
"""

import sys

import pandas as pd


def main():
    # File paths
    existing_file = "~/Downloads/NEW_all_eval_trail_benchmark - NEW_train_eval_trail_benchmark.csv"
    tool_judge_file = "tool_judges_0910_train.csv"
    output_file = "MERGED_TRAIN_TRAIL_BENCHMARK.csv"

    try:
        print(f"Reading existing benchmark data from {existing_file}...")
        # Read the existing benchmark data
        # Use quotechar and escapechar to handle the complex quoted content
        existing_df = pd.read_csv(existing_file, quotechar='"', escapechar=None)
        print(f"Loaded {len(existing_df)} rows from existing benchmark")

        print(f"Reading tool judge results from {tool_judge_file}...")

        tool_judge_df = pd.read_csv(
            tool_judge_file, quotechar='"', escapechar=None
        )

        tool_judge_df = tool_judge_df.assign(
            filename=tool_judge_df["filename"].str.replace(
                "/Users/dhuang/Documents/", ""
            )
        )

        print(f"Loaded {len(tool_judge_df)} rows from tool judge results")

        # Check if filename columns exist
        if "filename" not in existing_df.columns:
            raise ValueError(
                "'filename' column not found in existing benchmark data"
            )
        if "filename" not in tool_judge_df.columns:
            raise ValueError(
                "'filename' column not found in tool judge results"
            )

        # Check if tool judge columns exist
        required_cols = [
            "Tool Selection_score",
            "Tool Selection_reasons",
            "Tool Calling_score",
            "Tool Calling_reasons",
        ]
        for col in required_cols:
            if col not in tool_judge_df.columns:
                raise ValueError(
                    f"'{col}' column not found in tool judge results"
                )

        print("Merging data...")
        # Merge the dataframes on filename
        merged_df = existing_df.merge(
            tool_judge_df[["filename"] + required_cols],
            on="filename",
            how="left",
        )

        print(f"Merged dataset contains {len(merged_df)} rows")

        # Check for any tool judge results that didn't match existing data
        existing_filenames = set(existing_df["filename"])
        tool_judge_filenames = set(tool_judge_df["filename"])
        unmatched_tool_judge = tool_judge_filenames - existing_filenames
        if unmatched_tool_judge:
            print(
                f"Warning: {len(unmatched_tool_judge)} tool judge results have no matching existing data:"
            )
            for filename in sorted(
                list(unmatched_tool_judge)[:5]
            ):  # Show first 5
                print(f"  - {filename}")
            if len(unmatched_tool_judge) > 5:
                print(f"  ... and {len(unmatched_tool_judge) - 5} more")

        print(f"Saving merged results to {output_file}...")
        # Save the merged data
        merged_df.to_csv(
            output_file, index=False, quotechar='"', quoting=1
        )  # QUOTE_ALL

        print(f"✅ Successfully created {output_file}")
        print(f"   - Original data: {len(existing_df)} rows")
        print(f"   - Tool judge data: {len(tool_judge_df)} rows")
        print(f"   - Merged data: {len(merged_df)} rows")
        print(f"   - Columns in merged file: {len(merged_df.columns)}")

        # Show the new column names
        print("\nNew columns added:")
        for col in required_cols:
            print(f"   - {col}")

    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        print(
            "Make sure you're running this script from the directory containing the CSV files"
        )
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
