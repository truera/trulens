import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { TruSession } from "@trulens/core";
import { LangChainInstrumentation } from "@trulens/instrumentation-langchain";

const ENDPOINT = process.env.TRULENS_ENDPOINT ?? "http://localhost:4318";

export async function initSession(): Promise<TruSession> {
  return TruSession.init({
    appName: "langchain-rag",
    appVersion: "v1",
    exporter: new OTLPTraceExporter({ url: `${ENDPOINT}/v1/traces` }),
    endpoint: ENDPOINT,
    instrumentations: [new LangChainInstrumentation()],
  });
}
