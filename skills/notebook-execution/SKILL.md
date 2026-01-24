---
skill_spec_version: 0.1.0
name: trulens-notebook-execution
version: 1.0.0
description: Execute and display Jupyter notebooks for TruLens demos and quickstarts
tags: [trulens, jupyter, notebook, execution, demo]
---

# TruLens Notebook Execution

Execute Jupyter notebooks, display progress to the user, and handle API key requirements.

## When to Use This Skill

Use this skill when:
- Running TruLens quickstart or example notebooks
- Demonstrating TruLens functionality via notebooks
- Testing notebook examples end-to-end
- User asks to "run the notebook" or "execute the notebook"

## Execution Method

**Always use `jupyter nbconvert --execute`** to run notebooks. This:
- Maintains state across cells (variables persist)
- Captures all output properly
- Handles async operations correctly
- Works with OTEL tracing

**DO NOT** try to run notebooks by:
- Extracting cells and running them individually in bash
- Using `python -c` with heredocs
- Running as a standalone Python script (loses notebook context)

### Basic Execution Command

```bash
jupyter nbconvert --to notebook --execute --inplace <notebook_path>
```

### Execution with Timeout (for long-running notebooks)

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  <notebook_path>
```

### Execution with Output to stdout

```bash
jupyter nbconvert --to notebook --execute --stdout <notebook_path>
```

## Displaying Progress to User

When running a notebook, **display section headers as each cell executes** - NOT generic "BASH_OUTPUT" messages.

### Step 1: Parse the Notebook Structure First

Before executing, read the notebook JSON to build a map of:
- Markdown headers (## Section Name)
- Which code cells belong to which section

```python
import json

with open('notebook.ipynb') as f:
    nb = json.load(f)

sections = []
current_section = "Setup"

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        # Extract header
        for line in source.split('\n'):
            if line.startswith('## '):
                current_section = line.replace('## ', '').strip()
    sections.append((i, current_section))
```

### Step 2: Display Section Headers During Execution

When checking output or between cell groups, display the section name:

```
=== Step 1: Create the Search Tool ===
[cell output here]

=== Step 2: Create the Deep Agent ===
[cell output here]

=== Step 3: Set Up TruLens Session ===
[cell output here]
```

### Key Rule: Never Show "BASH_OUTPUT" to User

When polling for bash output during notebook execution:
- **DO**: Print the current section header before showing output
- **DON'T**: Just say "BASH_OUTPUT" or "checking output..."

### Example Display Implementation

```
# When starting a section:
print(f"\n=== {section_name} ===")

# When showing cell output:
print(output)

# When section completes:
print("✓ Complete")
```

### Progress Display Pattern

```
Running notebook: deep_agents_quickstart.ipynb

=== Step 1: Create the Search Tool ===
✓ Complete

=== Step 2: Create the Deep Agent ===
✓ Complete

=== Step 3: Set Up TruLens Session ===
Starting dashboard...
Dashboard started at http://localhost:8501
✓ Complete

=== Step 4: Define Agent GPA Feedback Functions ===
✓ Complete

=== Step 5: Instrument the Agent with TruGraph ===
✓ Complete

=== Step 6: Run and Evaluate ===
Running agent with question: "What is the weather in San Francisco?"
Agent response: "The weather in San Francisco is..."
Waiting for evaluation results...
✓ Evaluations complete

=== Results ===
Answer Relevance: 1.0
Tool Selection: 1.0
...
```

## Handling API Keys

**Critical: Check environment first, then prompt for keys ONE AT A TIME**

### Step 1: Check Environment

```bash
env | grep -E "OPENAI|TAVILY|ANTHROPIC" || echo "No API keys found"
```

### Step 2: If Keys Not Found, Prompt Individually

When prompting for keys:
- Ask for ONE key at a time
- Use the key prefix as a hint in the option label (e.g., "sk-proj-..." for OpenAI)
- Let users paste directly - don't rely on complex "Other" field workflows

Example prompt pattern:
```
Question: "Paste your OPENAI_API_KEY:"
Header: "OpenAI"
Options: [{"label": "sk-proj-...", "description": "Paste your sk-... key"}]
```

The user will paste their actual key by selecting "Other" or the option itself will be replaced with their input.

### Step 3: Set Keys When Running

```bash
OPENAI_API_KEY="sk-..." TAVILY_API_KEY="tvly-..." \
  jupyter nbconvert --execute ...
```

### Common API Keys for TruLens Notebooks

| Key | Used For |
|-----|----------|
| `OPENAI_API_KEY` | OpenAI LLM calls, embeddings, feedback provider |
| `TAVILY_API_KEY` | Web search tool (Deep Agents, research agents) |
| `ANTHROPIC_API_KEY` | Anthropic/Claude models |
| `HUGGINGFACE_API_KEY` | HuggingFace models |

## Keeping the Dashboard Alive

**Critical: The notebook execution process ends, killing any dashboard started within it.**

After notebook execution completes, launch the dashboard separately using TruLens's `run_dashboard()` function.

### Important: Database Location

The notebook writes its database to `./default.sqlite` **relative to the notebook's directory**. The `run_dashboard()` function reads from `./default.sqlite` relative to the **current working directory**.

**This means you MUST `cd` to the notebook's directory before launching the dashboard.**

### Correct Pattern for Dashboard Persistence

```bash
cd /path/to/notebook/directory && \
python3 << 'EOF'
from trulens.core import TruSession
from trulens.dashboard import run_dashboard

session = TruSession()
run_dashboard(session)
EOF
```

Use `run_in_background=true` with the bash tool so the dashboard stays alive.

### Why NOT to Use Native Streamlit Commands

**DO NOT** try to launch the dashboard with native streamlit commands like:
```bash
# WRONG - will connect to wrong/empty database!
streamlit run /path/to/trulens/src/dashboard/trulens/dashboard/main.py
```

This fails because:
1. Streamlit runs from the current working directory (likely repo root)
2. It looks for `./default.sqlite` relative to that directory
3. The actual database is in the notebook's directory
4. Result: "No apps found" in the dashboard

### Full Example Workflow

```bash
# Step 1: Execute notebook
OPENAI_API_KEY="sk-..." jupyter nbconvert --execute --inplace \
  /path/to/examples/notebook.ipynb

# Step 2: Launch persistent dashboard FROM THE NOTEBOOK'S DIRECTORY
cd /path/to/examples && \
python3 << 'EOF'
from trulens.core import TruSession
from trulens.dashboard import run_dashboard

session = TruSession()
run_dashboard(session)
EOF
# Use run_in_background=true for this command
```

The dashboard will output its URL (e.g., `http://localhost:55872`) and remain running until explicitly stopped.

## Post-Execution

After notebook execution:

1. **Show the leaderboard/results** if available
2. **Provide the dashboard URL** if one was launched
3. **Summarize what was evaluated** (metrics used, scores achieved)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Notebook times out | Increase timeout: `--ExecutePreprocessor.timeout=1200` |
| Kernel not found | Ensure correct Python environment is active |
| Import errors | Run `pip install` cell first or install dependencies |
| API key errors | Verify keys are set correctly in environment |
| Dashboard doesn't start | Check if port is already in use |
| `'id' was unexpected` error | Remove `id` fields from cells (see fix below) |

### Fixing Invalid Notebook JSON

If you see `Additional properties are not allowed ('id' was unexpected)`:

```python
import json

with open('notebook.ipynb', 'r') as f:
    nb = json.load(f)

# Remove 'id' fields from cells (not valid in nbformat 4)
for cell in nb['cells']:
    if 'id' in cell:
        del cell['id']

with open('notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
```

## Example Workflow

```
User: "Run the deep agents quickstart notebook"

1. Read notebook to identify:
   - Section headers (for progress display)
   - Required API keys (OPENAI_API_KEY, TAVILY_API_KEY)

2. Check environment for existing keys:
   env | grep -E "OPENAI|TAVILY"

3. Prompt for missing keys (ONE AT A TIME):
   "Please provide your OPENAI_API_KEY:"
   [User enters key]

   "Please provide your TAVILY_API_KEY:"
   [User enters key]

4. Execute notebook, displaying section headers:

   === Step 1: Create the Search Tool ===
   ✓ Complete

   === Step 2: Create the Deep Agent ===
   ✓ Complete

   === Step 3: Set Up TruLens Session ===
   ✓ Complete

   === Step 4: Define Agent GPA Feedback Functions ===
   ✓ Complete

   === Step 5: Instrument the Agent ===
   ✓ Complete

   === Step 6: Run and Evaluate ===
   Running agent...
   Waiting for evaluation results...
   ✓ Complete

5. Launch dashboard in background FROM THE NOTEBOOK'S DIRECTORY:
   cd /path/to/notebook/directory && python3 -c "
   from trulens.core import TruSession
   from trulens.dashboard import run_dashboard
   session = TruSession()
   run_dashboard(session)
   "
   [run_in_background=true]

6. Display results summary:
   "✓ Notebook execution complete!

    Evaluation Results:
    - Answer Relevance: 1.0
    - Tool Selection: 1.0
    - Tool Calling: 1.0
    - Execution Efficiency: 0.33
    - Plan Quality: 1.0
    - Plan Adherence: 1.0

    Dashboard running at: http://localhost:8501
    (Dashboard will stay alive until you stop it)"
```

## Integration with Other Skills

This skill works alongside:
- `instrumentation/` - for understanding what's being traced
- `evaluation-setup/` - for understanding feedback functions
- `running-evaluations/` - for interpreting results
