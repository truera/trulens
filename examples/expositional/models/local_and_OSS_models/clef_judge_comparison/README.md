# Local Clef versus Claude judges

Open [the notebook](clef_judge_comparison.ipynb) for the worked example and saved
outputs. It compares local Ollama Clef with Opus 5.5 and Sonnet 5.5 through
Snowflake Cortex, using public SummEval expert consistency labels.

The workflow uses native TruLens APIs:

- `Metric` wraps each judge.
- `Jury.repeated` runs repeated judgments and returns `reliability.*` metadata.
- `AlignmentReport` supplies human-agreement metrics, calibration curves,
  score distributions, confusion matrices, and worst misses.
- `CrossModelAlignmentReport` compares aligned scores across judges.
- `Jury` evaluates a median panel using the saved scores.

`decision_judges.py` contains only the protocol adapters. Clef uses Ollama's
System One API instead of chat completions. Claude uses Cortex Messages because
the native Cortex provider's legacy completion path returned HTTP 400 for
these models in this environment. Its scoring still delegates to TruLens
`LLMProvider.generate_score`; no custom rating parser is needed.

## Run

Install `requirements.txt` in your notebook kernel. Use current TruLens with
`Jury.repeated` support (or install the repository's packages in editable mode).
For local inference, install Ollama 0.35.1 or later and run `ollama pull clef`.
The download is approximately 18 GB; runtime memory needs are higher.

Leave `RUN_LIVE = False` to reproduce the saved analysis without network calls.
To collect new judgments, configure a named Snowflake connector connection,
set `SNOWFLAKE_CONNECTION_NAME`, and change `RUN_LIVE = True`. If your connection
uses an underscore hostname, set `SNOWFLAKE_HOST` to its TLS-valid account URL
hostname from Snowsight; TLS verification stays enabled. The default live
run makes 900 judgments plus one local warm-up. It incurs Snowflake charges.
Use a small `N_EXAMPLES` for a smoke test before running the full set.

There is no automatic model download, hosted Clef fallback, or retry. Notebook
live runs save `live_results.json` separately from the included snapshot. The
saved results were collected with an earlier interleaved runner and a strict
terminal-JSON parser; the simplified live path uses native parsing and serial
`Jury.repeated` calls. The notebook records this distinction and does not merge
the two configurations.

## Interpret results

The snapshot covers 100 held-out summaries from article-disjoint splits and
three calls per model. The dataset revision and checksum, rubric, model digest,
quantization, inference settings, and per-call measurements accompany it. Only
sanitized public-data results are published; no connection names, account
hostnames, request IDs, API keys, or raw trace databases are included.

SummEval labels are ordinal ratings normalized to 0–1. Calibration plots measure
score alignment, not calibrated event probabilities. No calibrator is fitted to
held-out labels. Multiple summaries per article and repeated calls are dependent;
these descriptive results do not establish statistical significance or universal
judge rankings. A zero repeat flip rate is not proof of general stability.

Latency includes client/network time. Local warm-up is excluded, and local
quantization/hardware affect performance. Cortex token usage is reported, but
its dollar cost is unknown. Clef has no provider API charge; local hardware and
electricity are not free and are not measured here.

## Sources

- [SummEval dataset](https://huggingface.co/datasets/mteb/summeval)
- [Ollama Clef](https://ollama.com/library/clef)
- [Ollama System One API](https://docs.ollama.com/api/systemone)
- [Snowflake Cortex REST API](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api)
