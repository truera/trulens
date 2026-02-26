/**
 * TruLens session setup for the demo.
 *
 * Uses the built-in SQLiteConnector so tracing works without a
 * separate Python receiver process.  The dashboard and evaluations
 * can read from the same `default.sqlite` file.
 */

import { TruSession, SQLiteConnector } from "@trulens/core";
import { OpenAIInstrumentation } from "@trulens/instrumentation-openai";

export async function initSession(): Promise<TruSession> {
  return TruSession.init({
    appName: process.env["APP_NAME"] ?? "trulens-rag-demo",
    appVersion: process.env["APP_VERSION"] ?? "v1",
    connector: new SQLiteConnector({
      dbPath: process.env["TRULENS_DB_PATH"] ?? "default.sqlite",
    }),
    instrumentations: [new OpenAIInstrumentation()],
  });
}
