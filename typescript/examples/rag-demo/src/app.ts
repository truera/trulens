/**
 * Entry point for the TruLens TypeScript SDK demo.
 *
 * Interactive REPL — type questions and see traced RAG answers.
 * Spans are flushed to the configured TruLens OTLP receiver on exit.
 *
 * Run with:
 *   OPENAI_API_KEY=sk-... pnpm start
 */

import * as readline from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { createTruApp } from "@trulens/core";
import { initSession } from "./trulens.js";
import { SimpleRAG } from "./rag.js";

async function main() {
  const session = await initSession();
  console.log(
    `TruSession initialised — app="${session.appName}" version="${session.appVersion}"`,
  );
  console.log("Spans written to", process.env["TRULENS_DB_PATH"] ?? "default.sqlite");
  console.log("Type a question and press Enter. Ctrl+C or empty line to quit.\n");

  const rag = new SimpleRAG();
  const app = createTruApp(rag, {
    mainMethod: "query",
    mainInput: (question: string) => question,
  });

  const rl = readline.createInterface({ input: stdin, output: stdout });

  try {
    while (true) {
      const question = await rl.question("Q: ");
      if (!question.trim()) break;

      try {
        const answer = await app.query(question);
        console.log(`A: ${answer}\n`);
      } catch (err) {
        console.error("Error:", err, "\n");
      }
    }
  } finally {
    rl.close();
    console.log("\nFlushing spans to TruLens…");
    await session.shutdown();
    console.log("Done. Open the TruLens dashboard to see the traces.");
  }
}

main()
  .catch((err) => {
    console.error(err);
    process.exit(1);
  })
  .finally(() => process.exit(0));
