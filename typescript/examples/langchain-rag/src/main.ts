import * as readline from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { initSession } from "./trulens.js";
import { buildChain } from "./rag.js";
import { createTruApp } from "@trulens/core";

async function main() {
  const session = await initSession();
  console.log(`TruSession ready — app="${session.appName}"`);
  console.log("Spans written to", process.env["TRULENS_DB_PATH"] ?? "default.sqlite");

  const chain = await buildChain();
  console.log("LangChain RAG chain built.\n");

  const app = createTruApp(
    { query: (q: string) => chain.invoke(q) },
    { mainMethod: "query", mainInput: (q: string) => q },
  );

  console.log("Type a question and press Enter. Empty line or Ctrl+C to quit.\n");
  const rl = readline.createInterface({ input: stdin, output: stdout });

  try {
    while (true) {
      const question = await rl.question("Q: ");
      if (!question.trim()) break;
      console.log(`A: ${await app.query(question)}\n`);
    }
  } finally {
    rl.close();
    console.log("\nFlushing spans…");
    await session.shutdown();
    console.log("Done.");
  }
}

main()
  .catch(console.error)
  .finally(() => process.exit(0));
