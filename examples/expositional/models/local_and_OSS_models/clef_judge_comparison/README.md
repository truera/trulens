# Local Clef versus Claude judges

Open [the notebook](clef_judge_comparison.ipynb) for the worked comparison and
saved outputs. It asks whether a local decision model can replace a frontier
LLM judge on summary factual consistency, comparing Ollama-hosted Clef (27B,
Q4_K_M) with Claude Opus 5.5 and Sonnet 5.5 through Snowflake Cortex on public
SummEval expert consistency labels.

## What the bundled run found

For **offline scoring**, the Claude judges ranked summaries more like the
experts: MAE 0.087 (Opus) and 0.089 (Sonnet) against 0.134 for Clef, with
Spearman 0.653 and 0.625 against 0.525. For **online guardrails**, Clef is the
better-shaped candidate: priced at published list rates it costs about 20-30x
less per judgment, it returns a typed score that cannot arrive unparsable, it
completed all 300 calls, and its count-weighted calibration error (0.074) is
close to Claude's (0.060 and 0.064).

The catch is latency in this deployment: 4.55 s median locally, too slow to
block a live request. Cloudflare reports 209 ms for hosted Clef and 38.8 ms for
Clef-flash, so measure p95 on the deployment you intend to run.

## Native TruLens APIs used

- `Metric` wraps each judge.
- `Jury.repeated` runs repeated judgments and returns `reliability.*` metadata.
- `AlignmentReport` supplies human-agreement metrics, calibration curves,
  score distributions, confusion matrices, and worst misses.
- `CrossModelAlignmentReport` compares aligned scores across judges.
- `Jury` evaluates a median panel using the saved scores.
- `block_output` turns the recorded scores into an online guardrail.

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
the two configurations. The written findings describe the bundled snapshot, so
revise them if you re-measure.

## Interpret results

The snapshot covers 100 held-out summaries from article-disjoint splits and
three calls per model. The dataset revision and checksum, rubric, model digest,
quantization, inference settings, and per-call measurements accompany it. Only
sanitized public-data results are published; no connection names, account
hostnames, request IDs, API keys, or raw trace databases are included.

SummEval labels are ordinal ratings normalized to 0-1, and 77 of the 100
summaries carry the top rating, so read MAE alongside rank correlation and the
worst misses. Calibration here means agreement with expert ratings, not
calibrated event probabilities. No calibrator is fitted to held-out labels and
the 0.75 threshold is the rubric's, not a tuned operating point. Multiple
summaries per article and repeated calls are dependent; these descriptive
results do not establish statistical significance or universal judge rankings.
A zero repeat flip rate is not proof of general stability.

Costs in the notebook are **estimates**: published list rates checked
2026-10-08 applied to measured token counts. Our Claude calls billed through
Snowflake Cortex at rates we did not obtain, hosted Clef rates differ by
provider, and local inference carries unmeasured hardware and electricity cost.
Latency includes client and network time and reflects this hardware and
quantization. Vendor latency and benchmark figures quoted for Clef, Clef-flash,
and Jev are not reproduced here.

## Sources

- [SummEval dataset](https://huggingface.co/datasets/mteb/summeval)
- [Ollama Clef](https://ollama.com/library/clef)
- [Ollama System One API](https://docs.ollama.com/api/systemone)
- [Cloudflare Workers AI pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/)
- [Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- [Snowflake Cortex REST API](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api)
