# TruLens TypeScript SDK — RAG Demo

A minimal Retrieval-Augmented Generation (RAG) app that demonstrates
end-to-end TruLens TypeScript instrumentation.

## What it shows

- `instrument()` wrapping `retrieve()` with `RETRIEVAL` span type + attributes
- `instrument()` wrapping `generate()` with `GENERATION` span type
- `withRecord()` creating the top-level `RECORD_ROOT` span
- Spans exported to a Python `TruSession` over OTLP/HTTP
- Traces visible in the TruLens Streamlit dashboard with no dashboard changes

## Run it

See the [top-level TypeScript README](../../README.md#running-the-demo) for
step-by-step instructions.

```bash
# Quick start (assumes Python TruSession OTLP receiver is running on :4318)
OPENAI_API_KEY=sk-... pnpm start
```
