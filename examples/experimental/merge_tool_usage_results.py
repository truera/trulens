#!/usr/bin/env python3
"""
Script to merge Tool Usage judge results with existing benchmark data.

This script reads the existing benchmark CSV and the new tool usage results CSV,
then creates a new merged CSV file with Tool Usage_score and Tool Usage_reasons
columns appended to the existing data.
"""

import sys

import pandas as pd


def main():
    # File paths
    existing_file = "EXISTING_TRAIN_TRAIL_BENCHMARK.csv"
    tool_usage_file = "tool_usage_0907_train.csv"
    output_file = "MERGED_TRAIN_TRAIL_BENCHMARK.csv"

    try:
        print(f"Reading existing benchmark data from {existing_file}...")
        # Read the existing benchmark data
        # Use quotechar and escapechar to handle the complex quoted content
        existing_df = pd.read_csv(existing_file, quotechar='"', escapechar=None)
        print(f"Loaded {len(existing_df)} rows from existing benchmark")

        print(f"Reading tool usage results from {tool_usage_file}...")
        # Read the tool usage results
        tool_usage_df = pd.read_csv(
            tool_usage_file, quotechar='"', escapechar=None
        )
        print(f"Loaded {len(tool_usage_df)} rows from tool usage results")

        # Check if filename columns exist
        if "filename" not in existing_df.columns:
            raise ValueError(
                "'filename' column not found in existing benchmark data"
            )
        if "filename" not in tool_usage_df.columns:
            raise ValueError(
                "'filename' column not found in tool usage results"
            )

        # Check if tool usage columns exist
        required_cols = ["Tool Usage_score", "Tool Usage_reasons"]
        for col in required_cols:
            if col not in tool_usage_df.columns:
                raise ValueError(
                    f"'{col}' column not found in tool usage results"
                )

        print("Merging data...")
        # Merge the dataframes on filename
        merged_df = existing_df.merge(
            tool_usage_df[["filename"] + required_cols],
            on="filename",
            how="left",
        )

        print(f"Merged dataset contains {len(merged_df)} rows")

        # Check for any missing matches
        missing_tool_usage = merged_df["Tool Usage_score"].isna().sum()
        if missing_tool_usage > 0:
            print(
                f"Warning: {missing_tool_usage} rows from existing data have no matching tool usage results"
            )

        # Check for any tool usage results that didn't match existing data
        existing_filenames = set(existing_df["filename"])
        tool_usage_filenames = set(tool_usage_df["filename"])
        unmatched_tool_usage = tool_usage_filenames - existing_filenames
        if unmatched_tool_usage:
            print(
                f"Warning: {len(unmatched_tool_usage)} tool usage results have no matching existing data:"
            )
            for filename in sorted(
                list(unmatched_tool_usage)[:5]
            ):  # Show first 5
                print(f"  - {filename}")
            if len(unmatched_tool_usage) > 5:
                print(f"  ... and {len(unmatched_tool_usage) - 5} more")

        print(f"Saving merged results to {output_file}...")
        # Save the merged data
        merged_df.to_csv(
            output_file, index=False, quotechar='"', quoting=1
        )  # QUOTE_ALL

        print(f"✅ Successfully created {output_file}")
        print(f"   - Original data: {len(existing_df)} rows")
        print(f"   - Tool usage data: {len(tool_usage_df)} rows")
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
