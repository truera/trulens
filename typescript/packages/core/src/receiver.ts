/**
 * TruLensReceiver — lightweight OTLP/HTTP receiver for TypeScript.
 *
 * Accepts OTLP JSON span payloads on POST /v1/traces and app
 * registration on POST /v1/register, then delegates persistence to a
 * `DBConnector`.  This is the TypeScript equivalent of the Python
 * `OTLPReceiver` in `trulens.core.otel.otlp_receiver`.
 */

import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";

import { ResourceAttributes } from "@trulens/semconv";

import { computeAppId } from "./app-id.js";
import type { DBConnector, EventRecord } from "./db-connector.js";

// ------------------------------------------------------------------ //
// OTLP JSON value extraction (mirrors Python's _json_any_value)
// ------------------------------------------------------------------ //

type AnyValue = Record<string, unknown>;

function jsonAnyValue(val: AnyValue): unknown {
  if ("stringValue" in val) return val.stringValue;
  if ("intValue" in val) {
    const raw = val.intValue;
    return typeof raw === "string" ? parseInt(raw, 10) : raw;
  }
  if ("doubleValue" in val) return val.doubleValue;
  if ("boolValue" in val) return val.boolValue;
  if ("arrayValue" in val) {
    const arr = val.arrayValue as { values?: AnyValue[] };
    return (arr.values ?? []).map(jsonAnyValue);
  }
  if ("kvlistValue" in val) {
    const kv = val.kvlistValue as { values?: { key: string; value: AnyValue }[] };
    const obj: Record<string, unknown> = {};
    for (const entry of kv.values ?? []) {
      obj[entry.key] = jsonAnyValue(entry.value);
    }
    return obj;
  }
  return null;
}

// ------------------------------------------------------------------ //
// Nano-timestamp → Date
// ------------------------------------------------------------------ //

function nanoToDate(ts: unknown): Date {
  if (ts == null) return new Date();
  const n = typeof ts === "string" ? BigInt(ts) : BigInt(ts as number);
  return new Date(Number(n / 1_000_000n));
}

// ------------------------------------------------------------------ //
// TruLensReceiver
// ------------------------------------------------------------------ //

export interface TruLensReceiverOptions {
  /** The DB connector to persist spans and apps. */
  connector: DBConnector;
  /** Port to listen on. Defaults to 4318. */
  port?: number;
  /** Host to bind to. Defaults to `"127.0.0.1"`. */
  host?: string;
}

export class TruLensReceiver {
  private readonly connector: DBConnector;
  private readonly port: number;
  private readonly host: string;
  private server: Server | null = null;

  constructor(options: TruLensReceiverOptions) {
    this.connector = options.connector;
    this.port = options.port ?? 4318;
    this.host = options.host ?? "127.0.0.1";
  }

  /** Start the HTTP server. Resolves once the server is listening. */
  start(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.server = createServer((req, res) => this._handleRequest(req, res));
      this.server.on("error", reject);
      this.server.listen(this.port, this.host, () => {
        console.log(
          `TruLensReceiver listening on http://${this.host}:${this.port}`,
        );
        resolve();
      });
    });
  }

  /** Stop the HTTP server and close the DB connector. */
  stop(): Promise<void> {
    return new Promise((resolve) => {
      if (this.server) {
        this.server.close(() => resolve());
      } else {
        resolve();
      }
    });
  }

  // ---------------------------------------------------------------- //
  // Request routing
  // ---------------------------------------------------------------- //

  private _handleRequest(req: IncomingMessage, res: ServerResponse): void {
    if (req.method !== "POST") {
      res.writeHead(405);
      res.end();
      return;
    }

    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      const body = Buffer.concat(chunks);
      try {
        if (req.url === "/v1/traces") {
          this._handleTraces(body, res);
        } else if (req.url === "/v1/register") {
          this._handleRegister(body, res);
        } else {
          res.writeHead(404);
          res.end();
        }
      } catch (err) {
        console.error("TruLensReceiver error:", err);
        res.writeHead(500);
        res.end(JSON.stringify({ error: String(err) }));
      }
    });
  }

  // ---------------------------------------------------------------- //
  // POST /v1/register
  // ---------------------------------------------------------------- //

  private _handleRegister(body: Buffer, res: ServerResponse): void {
    const payload = JSON.parse(body.toString("utf-8")) as {
      app_name: string;
      app_version?: string;
    };
    const appName = payload.app_name;
    const appVersion = payload.app_version ?? "base";
    const appId = computeAppId(appName, appVersion);

    this.connector.addApp({
      appId,
      appName,
      appVersion,
      appJson: {
        app_id: appId,
        app_name: appName,
        app_version: appVersion,
        metadata: {},
        tags: "-",
        record_ingest_mode: "immediate",
      },
    });

    this._sendJson(res, 200, { app_id: appId });
  }

  // ---------------------------------------------------------------- //
  // POST /v1/traces
  // ---------------------------------------------------------------- //

  private _handleTraces(body: Buffer, res: ServerResponse): void {
    const spans = this._decodeSpans(body);
    const events = this._spansToEvents(spans);

    if (events.length > 0) {
      this.connector.addEvents(events);
    }

    this._sendJson(res, 200, {});
  }

  // ---------------------------------------------------------------- //
  // OTLP JSON decoding
  // ---------------------------------------------------------------- //

  private _decodeSpans(body: Buffer): SpanDict[] {
    const payload = JSON.parse(body.toString("utf-8")) as OtlpPayload;
    const spans: SpanDict[] = [];

    for (const resourceSpans of payload.resourceSpans ?? []) {
      const resourceAttrs: Record<string, unknown> = {};
      for (const kv of resourceSpans.resource?.attributes ?? []) {
        resourceAttrs[kv.key] = jsonAnyValue(kv.value);
      }
      for (const scopeSpans of resourceSpans.scopeSpans ?? []) {
        for (const span of scopeSpans.spans ?? []) {
          const attrs: Record<string, unknown> = {};
          for (const kv of span.attributes ?? []) {
            attrs[kv.key] = jsonAnyValue(kv.value);
          }
          spans.push({
            name: span.name,
            traceId: span.traceId,
            spanId: span.spanId,
            parentSpanId: span.parentSpanId ?? null,
            startTimeUnixNano: span.startTimeUnixNano,
            endTimeUnixNano: span.endTimeUnixNano,
            attributes: attrs,
            resourceAttributes: resourceAttrs,
            status: span.status ?? {},
          });
        }
      }
    }
    return spans;
  }

  // ---------------------------------------------------------------- //
  // Span → EventRecord transformation
  // (port of Python _ingest_spans, otlp_receiver.py:279-350)
  // ---------------------------------------------------------------- //

  private _spansToEvents(spans: SpanDict[]): EventRecord[] {
    const events: EventRecord[] = [];

    for (const s of spans) {
      const attrs = { ...s.attributes };
      const resAttrs = { ...s.resourceAttributes };

      // Copy app identity from span attrs into resource attrs
      // (mirrors the Python workaround)
      for (const k of [
        ResourceAttributes.APP_ID,
        ResourceAttributes.APP_NAME,
        ResourceAttributes.APP_VERSION,
      ]) {
        if (k in attrs) {
          resAttrs[k] = attrs[k];
        }
      }

      // Skip non-TruLens spans
      if (!resAttrs[ResourceAttributes.APP_NAME]) continue;

      const spanId = s.spanId ?? "";
      const parentId = s.parentSpanId ?? "";
      const traceId = s.traceId ?? "";

      const statusCode =
        typeof s.status.code === "number" ? s.status.code : 0;
      const statusStr =
        statusCode === 2 ? "STATUS_CODE_ERROR" : "STATUS_CODE_UNSET";

      events.push({
        eventId: spanId,
        record: {
          name: s.name ?? "",
          kind: 1, // SPAN_KIND_INTERNAL
          parent_span_id: parentId,
          status: statusStr,
        },
        recordAttributes: attrs,
        recordType: "SPAN",
        resourceAttributes: resAttrs,
        startTimestamp: nanoToDate(s.startTimeUnixNano),
        timestamp: nanoToDate(s.endTimeUnixNano),
        trace: {
          span_id: spanId,
          trace_id: traceId,
          parent_id: parentId,
        },
      });
    }

    return events;
  }

  // ---------------------------------------------------------------- //
  // Helpers
  // ---------------------------------------------------------------- //

  private _sendJson(
    res: ServerResponse,
    code: number,
    obj: Record<string, unknown>,
  ): void {
    res.writeHead(code, { "Content-Type": "application/json" });
    res.end(JSON.stringify(obj));
  }
}

// ------------------------------------------------------------------ //
// Type helpers for OTLP JSON payload
// ------------------------------------------------------------------ //

interface OtlpKeyValue {
  key: string;
  value: AnyValue;
}

interface OtlpSpan {
  name?: string;
  traceId?: string;
  spanId?: string;
  parentSpanId?: string;
  startTimeUnixNano?: string | number;
  endTimeUnixNano?: string | number;
  attributes?: OtlpKeyValue[];
  status?: Record<string, unknown>;
}

interface OtlpScopeSpans {
  spans?: OtlpSpan[];
}

interface OtlpResourceSpans {
  resource?: { attributes?: OtlpKeyValue[] };
  scopeSpans?: OtlpScopeSpans[];
}

interface OtlpPayload {
  resourceSpans?: OtlpResourceSpans[];
}

interface SpanDict {
  name: string | undefined;
  traceId: string | undefined;
  spanId: string | undefined;
  parentSpanId: string | null;
  startTimeUnixNano: string | number | undefined;
  endTimeUnixNano: string | number | undefined;
  attributes: Record<string, unknown>;
  resourceAttributes: Record<string, unknown>;
  status: Record<string, unknown>;
}
