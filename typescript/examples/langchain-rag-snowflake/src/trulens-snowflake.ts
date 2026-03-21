import { TruSession } from "@trulens/core";
import { LangChainInstrumentation } from "@trulens/instrumentation-langchain";
import {
  TruLensSnowflakeSpanExporter,
  SnowflakeRunManager,
} from "@trulens/connectors-snowflake";
import { sfConnector } from "./snowflake-connection.js";

const APP_NAME = "langchain-typescript-rag";
const APP_VERSION = "v1";
export const RUN_NAME = `langchain_ts_rag_run_${Date.now()}`;

export const runManager = new SnowflakeRunManager({
  connector: sfConnector,
});

export async function initSession(): Promise<TruSession> {
  return TruSession.init({
    appName: APP_NAME,
    appVersion: APP_VERSION,
    runName: RUN_NAME,
    exporter: new TruLensSnowflakeSpanExporter({
      connector: sfConnector,
    }),
    instrumentations: [new LangChainInstrumentation()],

    onInit: async () => {
      const objectName = await runManager.ensureExternalAgent(
        APP_NAME,
        APP_VERSION,
      );
      console.log(`External Agent: ${objectName}`);

      await runManager.createRun({
        objectName: APP_NAME,
        appVersion: APP_VERSION,
        runName: RUN_NAME,
        datasetName: "langchain_ts_rag_queries",
        sourceType: "DATAFRAME",
      });
      console.log(`Run created: ${RUN_NAME}`);
    },

    onShutdown: async (count) => {
      console.log(
        `Finalizing run (${count} records traced)...`,
      );
      await runManager.finalizeRun({
        objectName: APP_NAME,
        appVersion: APP_VERSION,
        runName: RUN_NAME,
        inputRecordsCount: count,
      });
    },
  });
}
