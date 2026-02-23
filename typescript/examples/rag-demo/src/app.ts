/**
 * Entry point for the TruLens TypeScript SDK demo.
 *
 * Runs a handful of questions through SimpleRAG and flushes all spans to
 * the configured TruLens OTLP receiver (Python TruSession) or Snowflake.
 *
 * Run with:
 *   OPENAI_API_KEY=sk-... pnpm start
 */

import { initSession } from "./trulens.js";
import { SimpleRAG } from "./rag.js";

const QUESTIONS = [
  "What is TruLens and what does it do?",
  "How does RAG work?",
  "What is OpenTelemetry?",
];

async function main() {
  // Initialise TruSession before any instrumented code runs.
  const session = initSession();
  console.log(
    `TruSession initialised — app="${session.appName}" version="${session.appVersion}"`
  );
  console.log(
    "Spans will be exported to TRULENS_OTLP_ENDPOINT =",
    process.env["TRULENS_OTLP_ENDPOINT"] ?? "http://localhost:4318"
  );
  console.log();

  const rag = new SimpleRAG();

  for (const question of QUESTIONS) {
    console.log(`Q: ${question}`);
    try {
      const answer = await rag.query(question);
      console.log(`A: ${answer}`);
    } catch (err) {
      console.error(`Error answering "${question}":`, err);
    }
    console.log();
  }

  // Flush all buffered spans before the process exits.
  console.log("Flushing spans to TruLens…");
  await session.shutdown();
  console.log("Done. Open the TruLens dashboard to see the traces.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
