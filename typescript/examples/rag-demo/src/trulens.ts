/**
 * TruLens session setup for the demo.
 *
 * Sends spans to a Python TruSession OTLP receiver running on localhost:4318.
 * Swap the exporter for TruLensSnowflakeSpanExporter to use the direct
 * Snowflake path instead (see README.md).
 */

import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { TruSession } from "@trulens/core";

const OTLP_ENDPOINT =
  process.env["TRULENS_OTLP_ENDPOINT"] ?? "http://localhost:4318";

export function initSession(): TruSession {
  const exporter = new OTLPTraceExporter({
    url: `${OTLP_ENDPOINT}/v1/traces`,
  });

  return TruSession.init({
    appName: process.env["APP_NAME"] ?? "trulens-rag-demo",
    appVersion: process.env["APP_VERSION"] ?? "v1",
    exporter,
  });
}
