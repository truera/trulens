/**
 * TruLens session setup for the LangChain RAG demo.
 *
 * Uses the built-in SQLiteConnector so tracing works without a
 * separate Python receiver process.
 */

import { TruSession, SQLiteConnector } from "@trulens/core";
import { LangChainInstrumentation } from "@trulens/instrumentation-langchain";

export async function initSession(): Promise<TruSession> {
  return TruSession.init({
    appName: "langchain-rag",
    appVersion: "v1",
    connector: new SQLiteConnector({
      dbPath: process.env["TRULENS_DB_PATH"] ?? "default.sqlite",
    }),
    instrumentations: [new LangChainInstrumentation()],
  });
}
