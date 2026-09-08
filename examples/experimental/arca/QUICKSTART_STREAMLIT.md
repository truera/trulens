# Quick Start Guide - Cortex Agent Prompt Optimizer

Get up and running with the Streamlit Agent Optimizer in 5 minutes!

## Prerequisites

1. **Snowflake Account** with:
   - At least one Cortex Agent created
   - Personal Access Token (PAT) with appropriate permissions
   - AI Observability enabled (optional, for GPA metrics)

2. **API Keys**:
   - OpenAI API key (for prompt generation and evaluation)

## Installation

```bash
# Navigate to the directory
cd /path/to/trulens/examples/experimental/arca

# Install required packages
pip install streamlit pandas plotly requests python-dotenv openai snowflake-snowpark-python

# Optional: Install TruLens for advanced evaluations
pip install trulens-core trulens-connectors-snowflake trulens-providers-openai
```

## Quick Setup

### 1. Configure Environment (Optional)

Create a `.env` file:

```bash
# .env
SNOWFLAKE_ACCOUNT_URL=https://your-account.snowflakecomputing.com
SNOWFLAKE_DATABASE=YOUR_DATABASE
SNOWFLAKE_SCHEMA=YOUR_SCHEMA
SNOWFLAKE_PAT=your_pat_token
SNOWFLAKE_USER=your_username
OPENAI_API_KEY=your_openai_key
```

Or enter them directly in the app sidebar.

### 2. Launch the App

```bash
streamlit run streamlit_agent_optimizer.py
```

Your browser should open automatically to `http://localhost:8501`

## First Run - Basic Workflow

### Step 1: Configure Connection (Sidebar)
- Enter your Snowflake Account URL
- Enter Database name
- Enter Schema name
- Enter your PAT (Personal Access Token)

### Step 2: Select an Agent
1. Click **"🔍 Load Agents"** button
2. Select your agent from the dropdown
3. Click **"📋 Load Agent Details"** to view current prompts

### Step 3: Load Test Data
1. Go to the **"📤 Upload CSV"** tab
2. Use the provided `sample_evaluation_data.csv` or upload your own
3. Preview the data
4. Click **"✅ Use This Dataset"**

### Step 4: Run Your Agent
1. Click **"▶️ Run Agent on Dataset"** (big green button)
2. Wait for completion (progress bar shows status)
3. Review the results table

### Step 5: Run Basic Evaluation
1. Scroll to **"5️⃣ LLM Judge Evaluation"**
2. Ensure **"Accuracy"** is checked
3. Click **"🧪 Run Evaluations"**
4. View your metrics!

### Step 6: Try Manual Optimization
1. Scroll to **"6️⃣ Prompt Optimization"**
2. Check **"Enable Manual Editing"**
3. Edit the system/response/orchestration prompts
4. Click **"✅ Apply Manual Prompts"**
5. Go back to Step 4 to test the new prompts

### Step 7: Try Automated Optimization
1. In **"6️⃣ Prompt Optimization"**, scroll to **"Automated Optimization"**
2. Set "Number of Iterations" to 2-3 (start small!)
3. Click **"🚀 Start Optimization"**
4. Watch as the app:
   - Generates new prompts
   - Tests them on your data
   - Evaluates results
   - Repeats!

### Step 8: Compare and Apply Best Results
1. Scroll to **"7️⃣ Results Comparison"**
2. View the optimization history table
3. Check the line chart showing metrics over iterations
4. The app shows you the **best iteration**
5. Click **"✅ Apply Best Prompts to Agent"** to update your agent

## Tips for Success

### Start Small
- Use a small dataset (10-20 questions) for your first runs
- Start with 2-3 optimization iterations
- Add more as you get comfortable

### Use Good Test Data
- Include diverse question types
- Provide ground truth when possible
- Cover edge cases and common scenarios

### Iterate Gradually
- Review results after each optimization
- Manual editing can complement automated optimization
- Don't over-optimize - watch for diminishing returns

### Monitor Feedback
- Use the optional human feedback feature
- Helps understand where agent struggles
- Can inform prompt improvements

## Common First-Time Issues

### "Failed to load agents"
**Solution**: Double-check your database and schema names match exactly (case-sensitive!)

### "TruLens not available"
**Solution**: This is OK! You can still use accuracy evaluation. To enable GPA metrics, install TruLens packages.

### "Failed to evaluate"
**Solution**: Make sure your OPENAI_API_KEY environment variable is set.

### Agent takes too long
**Solution**: Reduce dataset size for faster testing, or adjust agent's timeout settings.

## Example Session

Here's what a typical first session looks like:

1. **Connect** (30 seconds)
   - Enter credentials in sidebar
   - Load agents

2. **Setup** (1 minute)
   - Select agent
   - Upload sample_evaluation_data.csv
   - Preview everything looks good

3. **Baseline** (2 minutes)
   - Run agent on 16 questions
   - Run evaluation
   - Note baseline accuracy: 0.75

4. **First Optimization** (3 minutes)
   - Run 3 iterations of automated optimization
   - Watch metrics improve
   - Best iteration: accuracy 0.85

5. **Apply** (30 seconds)
   - Review best prompts
   - Apply to agent
   - Success! 🎉

**Total time: ~7 minutes**

## Next Steps

Once you're comfortable with the basics:

1. **Try larger datasets** (50-100+ questions)
2. **Enable GPA metrics** for deeper insights
3. **Experiment with temperature settings** for prompt generation
4. **Provide human feedback** to guide optimization
5. **Compare multiple optimization runs** with different strategies
6. **Export results** to share with your team

## Getting Help

- Check the full README_STREAMLIT.md for detailed documentation
- Review the code in streamlit_agent_optimizer.py
- Look at cortex_agent_manager.py to understand the API wrapper
- Check ray_optuna_cortex_test_optim.py for advanced optimization patterns

## Sample Data

The provided `sample_evaluation_data.csv` includes:
- Basic factual questions
- Math problems
- Reasoning tasks
- Contextual understanding
- Technical questions
- Historical facts

Perfect for testing general-purpose agents!

## Ready to Start?

```bash
streamlit run streamlit_agent_optimizer.py
```

Happy optimizing! 🚀

