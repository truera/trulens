import * as readline from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { sfConnector, connectAndVerify } from "./snowflake-connection.js";
import { initSession, runManager, RUN_NAME } from "./trulens-snowflake.js";
import { buildChain } from "./rag.js";
import { createTruApp } from "@trulens/core";

async function main() {
  await connectAndVerify();

  // Initialise TruSession (creates External Agent + Run)
  const session = await initSession();
  console.log(`TruSession ready — app="${session.appName}"`);

  const chain = await buildChain();
  console.log("LangChain RAG chain built.\n");

  const app = createTruApp(
    { query: (q: string) => chain.invoke(q) },
    { mainMethod: "query", mainInput: (q: string) => q },
  );

  console.log("Ask about the TruLens TypeScript SDK. Try:");
  console.log("  - How are spans serialized and uploaded to Snowflake?");
  console.log("  - How does record_id propagation work across batches?");
  console.log("  - What SQL calls create a Snowflake Run?");
  console.log("  - How does the LangChain tracer map run types to span types?");
  console.log("  - What happens during session.shutdown()?\n");
  const rl = readline.createInterface({ input: stdin, output: stdout });

  try {
    while (true) {
      const question = await rl.question("Q: ");
      if (!question.trim()) break;
      console.log(`A: ${await app.query(question)}\n`);
    }
  } finally {
    rl.close();
  }

  console.log("\nFlushing spans and finalizing run...");
  await session.shutdown();

  console.log("Waiting for Snowflake ingestion to complete...");
  await runManager.waitForIngestion({
    objectName: "langchain-typescript-rag",
    runName: RUN_NAME,
  });
  console.log("Ingestion complete.");

  console.log("Computing metrics: answer_relevance, context_relevance, groundedness...");
  await runManager.computeMetrics({
    objectName: "langchain-typescript-rag",
    appVersion: "v1",
    runName: RUN_NAME,
    metrics: ["answer_relevance", "context_relevance", "groundedness"],
  });

  console.log("Waiting for metric computation to complete...");
  const finalStatus = await runManager.waitForMetrics({
    objectName: "langchain-typescript-rag",
    runName: RUN_NAME,
  });
  console.log(`Metrics ${finalStatus.toLowerCase()}.`);

  await sfConnector.close();
  console.log("\nDone! Open Snowflake AI Observability UI to see traces and evaluations.");
}

main().catch(console.error);
