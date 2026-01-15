# GPA (GEPA Judge Optimization) Experiments

This directory contains experiments for optimizing LLM-based judges using **GEPA (Generative-Evolutionary Prompt Alignment)** from DSPy. The project focuses on evaluating and improving judge prompts for detecting errors in AI agent execution traces from two benchmarks: **GAIA** and **SWE-Bench**.

## Overview

The GPA experiments use a meta-judge approach to:
1. Analyze agent execution traces for specific error categories
2. Optimize judge prompts using GEPA to improve error detection recall
3. Evaluate optimized prompts against held-out test sets

### Error Categories

The system evaluates six error categories:
- **LC** - Logical Consistency
- **EE** - Execution Efficiency
- **PA** - Plan Adherence
- **PQ** - Plan Quality
- **TC** - Tool Calling
- **TS** - Tool Selection

## Directory Structure

```
GPA/
├── gepa_gaia.py                    # GEPA optimization script for GAIA benchmark
├── gepa_swebench.py                # GEPA optimization script for SWE-Bench benchmark
├── preprocess_trail_gaia.py        # Preprocessor to convert GAIA JSON traces to text
├── preprocess_trail_swebench.py    # Preprocessor to convert SWE-Bench JSON traces to text
├── trail_benchmark.ipynb           # Jupyter notebook for interactive benchmarking
│
├── GAIA/                           # GAIA trace data directory
│   ├── *.json                      # Raw JSON trace files from GAIA benchmark
│   └── *.txt                       # Preprocessed text traces (generated)
│
├── SWE_Bench/                      # SWE-Bench trace data directory
│   ├── *.json                      # Raw JSON trace files from SWE-Bench benchmark
│   └── *.txt                       # Preprocessed text traces (generated)
│
├── GPA Judge Error Analysis - TRAIN_CSV.csv   # GAIA training set with error annotations
├── GPA Judge Error Analysis - TEST_CSV.csv    # GAIA test set with error annotations
├── TRAIL_GAIA_Judge_Output_Per_Trace.csv      # GAIA judge outputs per trace
├── SWE-Bench_Train.csv                        # SWE-Bench training set with error annotations
├── SWE-Bench_Test.csv                         # SWE-Bench test set with error annotations
└── README.md                                  # This file
```

## Dependencies

### Python Packages

```bash
pip install dspy pandas trulens snowflake-snowpark-python scikit-learn
```

### Required Packages:
- `dspy` - For GEPA optimization and LLM orchestration
- `pandas` - Data manipulation and CSV handling
- `trulens` - TruLens feedback functions (specifically `trulens.feedback.v2`)
- `snowflake-snowpark-python` - Snowflake connection (for notebook)
- `scikit-learn` - Train/test splitting (for notebook)

### Environment Variables

The scripts require Snowflake credentials for LLM API access:

```bash
export SNOWFLAKE_ACCOUNT="your-snowflake-account"  # e.g., "SFCOGSOPS-SNOWHOUSE-AWS-US-WEST-2"
export SNOWFLAKE_JWT="your-jwt-token"
```

## Data Files

### CSV Files (Required)

| File | Description |
|------|-------------|
| `GPA Judge Error Analysis - TRAIN_CSV.csv` | GAIA training annotations with columns: `Filename`, `Align_Judges`, `Caught`, `Raw Error` |
| `GPA Judge Error Analysis - TEST_CSV.csv` | GAIA test annotations (same schema) |
| `SWE-Bench_Train.csv` | SWE-Bench training annotations with columns: `file`, `GPA Category (AJ)`, `error` |
| `SWE-Bench_Test.csv` | SWE-Bench test annotations (same schema) |

### Trace Files (Required)

- **GAIA/**: Contains `.json` raw traces and `.txt` preprocessed traces
- **SWE_Bench/**: Contains `.json` raw traces and `.txt` preprocessed traces

The `.txt` files are generated from `.json` files using the preprocessing scripts.

## Running the Scripts

### 1. Preprocess Trace Files

Before running optimization, convert JSON traces to readable text format:

**For GAIA traces:**
```bash
python preprocess_trail_gaia.py --input_file GAIA/<trace_id>.json --output_file GAIA/<trace_id>.txt
```

**For SWE-Bench traces:**
```bash
python preprocess_trail_swebench.py --input_file SWE_Bench/<trace_id>.json --output_file SWE_Bench/<trace_id>.txt
```

**Batch processing (via notebook):**
The `trail_benchmark.ipynb` notebook contains cells to batch-process all traces in each directory.

### 2. Running GEPA Optimization (gepa_gaia.py)

The `gepa_gaia.py` script optimizes judge prompts for the GAIA benchmark.

**Configuration (edit variables in script):**

```python
# Mode selection
RUN_OPTIMIZATION = True           # True = optimize prompts, False = evaluate existing prompts
OPTIMIZATION_CATEGORIES = ["PA"]  # Categories to optimize (empty list = all categories)

# Input/Output files
INPUT_PROMPT_FILE = "auto-medium_gaia_prompts.json"  # For evaluation mode
OUTPUT_PROMPT_FILE = "gaia_optimized_judge_prompts"  # Output prefix
LOG_FILE_PREFIX = "auto-medium-fixed-metajudge-gaia" # Log file prefix

# Parallelization
MAX_PARALLEL_WORKERS = 2  # Number of parallel category optimizations
GEPA_NUM_THREADS = 2      # Threads per GEPA optimizer

# Debug mode
TEST_SINGLE_EXAMPLE = False                        # Test single trace instead of full set
SINGLE_EXAMPLE_CATEGORY = "PA"                     # Category for single example test
SINGLE_EXAMPLE_FILE = "18efa24e637b9423f34180d1f2041d3e"  # Trace file for single test
```

**Run optimization:**
```bash
export SNOWFLAKE_ACCOUNT="your-account"
export SNOWFLAKE_JWT="your-jwt-token"
python gepa_gaia.py
```

**Outputs:**
- `gaia_optimized_judge_prompts_<timestamp>.json` - Optimized prompts per category
- `gaia_prompt_iterations_<timestamp>.json` - Starting and final prompts for comparison
- `auto-medium-fixed-metajudge-gaia_<timestamp>.log` - Detailed execution log

### 3. Running GEPA Optimization (gepa_swebench.py)

The `gepa_swebench.py` script optimizes judge prompts for the SWE-Bench benchmark.

**Configuration (edit variables in script):**

```python
# Mode selection
RUN_OPTIMIZATION = False          # True = optimize, False = evaluate
OPTIMIZATION_CATEGORIES = ["LC", "EE", "TC"]  # Categories to optimize

# Input/Output files
INPUT_PROMPT_FILE = "auto_light_swebench_optimized_judge_prompts_20251129_221543.json"
OUTPUT_PROMPT_FILE = "fixed_metajudge_auto_light_swebench_optimized_judge_prompts_run2"
LOG_FILE_PREFIX = "auto_light_swebench_metajudge_temp0_eval"

# Parallelization
MAX_PARALLEL_WORKERS = 2
GEPA_NUM_THREADS = 2

# Debug mode
TEST_SINGLE_EXAMPLE = False
SINGLE_EXAMPLE_CATEGORY = "LC"
SINGLE_EXAMPLE_FILE = "0e6f7928953ab5a568bae640ce915cc3.json"
```

**Run optimization:**
```bash
export SNOWFLAKE_ACCOUNT="your-account"
export SNOWFLAKE_JWT="your-jwt-token"
python gepa_swebench.py
```

**Outputs:**
- `fixed_metajudge_auto_light_swebench_optimized_judge_prompts_run2_<timestamp>.json`
- `fixed_metajudge_auto_light_swebench_prompt_iterations_run2_<timestamp>.json`
- `auto_light_swebench_metajudge_temp0_eval_<timestamp>.log`

### 4. Using the Jupyter Notebook (trail_benchmark.ipynb)

The notebook provides interactive benchmarking without GEPA optimization.

**Setup:**
1. Configure Snowflake connection in Cell 1:
   ```python
   snowflake_connection_parameters = {
       "account": "SNOWHOUSE",
       "user": "your-username",
       "authenticator": "externalbrowser",
   }
   ```

2. Run cells sequentially to:
   - Initialize TruLens feedback functions
   - Preprocess all trace files
   - Run evaluations on individual or all traces

**Key sections:**
- **Cells 1-2**: Snowflake + TruLens setup
- **Cells 4-7**: GAIA trace processing and custom instructions
- **Cells 9, 13**: Run GAIA evaluations (single/batch)
- **Cells 15-18**: SWE-Bench trace processing and custom instructions
- **Cells 20, 24**: Run SWE-Bench evaluations (single/batch)

## Workflow Summary

### Optimization Workflow

1. **Prepare data**: Ensure CSV annotations and JSON traces are in place
2. **Preprocess traces**: Run preprocessing scripts to generate `.txt` files
3. **Configure script**: Set `RUN_OPTIMIZATION = True` and choose categories
4. **Run optimization**: Execute `python gepa_gaia.py` or `python gepa_swebench.py`
5. **Review outputs**: Check generated JSON prompts and log files

### Evaluation Workflow

1. **Load existing prompts**: Set `RUN_OPTIMIZATION = False` and `INPUT_PROMPT_FILE`
2. **Run evaluation**: Execute the script
3. **Review metrics**: Check log file for recall scores per category

## Architecture Details

### GAIA Agent Traces

GAIA traces follow a hierarchical structure:
- **Manager Agent (CodeAgent)**: High-level planning and delegation
- **Search Agent (ToolCallingAgent)**: Executes tool-based subtasks

The preprocessing script (`preprocess_trail_gaia.py`) extracts LLM calls and organizes messages by agent type.

### SWE-Bench Agent Traces

SWE-Bench traces follow a simpler structure:
- **CodeAgent**: Single agent with access to sandbox, Python interpreter, and gitingest
- **Execution cycle**: Thought -> Code -> Observation sequences

The preprocessing script (`preprocess_trail_swebench.py`) extracts the execution flow and new messages per LLM call.

### GEPA Optimization Process

1. **Student Judge**: Analyzes traces using starting prompts from TruLens feedback classes
2. **Meta Judge**: Evaluates student output against ground-truth errors
3. **GEPA Optimizer**: Iteratively improves student prompts based on meta-judge feedback
4. **Validation**: Tests optimized prompts on held-out examples

## Troubleshooting

### Common Issues

1. **Missing JWT token**: Ensure `SNOWFLAKE_JWT` is exported
2. **Rate limiting (429 errors)**: Script has built-in retry with exponential backoff
3. **Missing trace files**: Run preprocessing scripts first
4. **CSV column mismatch**: Verify CSV columns match expected schema

### Debug Mode

Enable single-example testing to debug specific traces:
```python
TEST_SINGLE_EXAMPLE = True
SINGLE_EXAMPLE_CATEGORY = "LC"
SINGLE_EXAMPLE_FILE = "your_trace_file_id"
```

## File Dependencies Map

| Script | Required Files |
|--------|----------------|
| `gepa_gaia.py` | `GPA Judge Error Analysis - TRAIN_CSV.csv`, `GPA Judge Error Analysis - TEST_CSV.csv`, `TRAIL_GAIA_Judge_Output_Per_Trace.csv`, `GAIA/*.txt` |
| `gepa_swebench.py` | `SWE-Bench_Train.csv`, `SWE-Bench_Test.csv`, `SWE_Bench/*.txt` |
| `preprocess_trail_gaia.py` | `GAIA/*.json` |
| `preprocess_trail_swebench.py` | `SWE_Bench/*.json` |
| `trail_benchmark.ipynb` | All of the above |
