import { afterEach, describe, expect, it } from "vitest";
import http from "node:http";
import { TruLensReceiver } from "../receiver.js";
import type { AppDefinition, DBConnector, EventRecord } from "../db-connector.js";

// ---------------------------------------------------------------------------
// In-memory DBConnector stub
// ---------------------------------------------------------------------------

class StubConnector implements DBConnector {
  apps: AppDefinition[] = [];
  events: EventRecord[] = [];

  addApp(app: AppDefinition): string {
    this.apps.push(app);
    return app.appId;
  }

  addEvents(events: EventRecord[]): string[] {
    this.events.push(...events);
    return events.map((e) => e.eventId);
  }

  close(): void {}
}

// ---------------------------------------------------------------------------
// HTTP helper
// ---------------------------------------------------------------------------

function post(
  port: number,
  path: string,
  body: unknown
): Promise<{ status: number; body: string }> {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const req = http.request(
      {
        hostname: "127.0.0.1",
        port,
        path,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(data),
        },
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (c: Buffer) => chunks.push(c));
        res.on("end", () =>
          resolve({
            status: res.statusCode ?? 0,
            body: Buffer.concat(chunks).toString("utf-8"),
          })
        );
      }
    );
    req.on("error", reject);
    req.end(data);
  });
}

function get(
  port: number,
  path: string
): Promise<{ status: number; body: string }> {
  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname: "127.0.0.1",
        port,
        path,
        method: "GET",
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (c: Buffer) => chunks.push(c));
        res.on("end", () =>
          resolve({
            status: res.statusCode ?? 0,
            body: Buffer.concat(chunks).toString("utf-8"),
          })
        );
      }
    );
    req.on("error", reject);
    req.end();
  });
}

// ---------------------------------------------------------------------------
// Build a minimal OTLP JSON payload
// ---------------------------------------------------------------------------

function otlpPayload(
  spans: Array<{
    name?: string;
    traceId?: string;
    spanId?: string;
    parentSpanId?: string;
    attributes?: Array<{ key: string; value: Record<string, unknown> }>;
    startTimeUnixNano?: string;
    endTimeUnixNano?: string;
    status?: Record<string, unknown>;
  }>,
  resourceAttrs: Array<{ key: string; value: Record<string, unknown> }> = []
) {
  return {
    resourceSpans: [
      {
        resource: { attributes: resourceAttrs },
        scopeSpans: [{ spans }],
      },
    ],
  };
}

function strVal(v: string) {
  return { stringValue: v };
}

function intVal(v: number | string) {
  return { intValue: v };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

let receiver: TruLensReceiver | null = null;
let stub: StubConnector;
let port: number;

afterEach(async () => {
  if (receiver) {
    await receiver.stop();
    receiver = null;
  }
});

async function startReceiver(): Promise<void> {
  stub = new StubConnector();
  // Use port 0 to get a random available port
  port = 14318 + Math.floor(Math.random() * 10000);
  receiver = new TruLensReceiver({
    connector: stub,
    port,
    host: "127.0.0.1",
  });
  await receiver.start();
}

describe("TruLensReceiver", () => {
  describe("HTTP routing", () => {
    it("returns 405 for non-POST methods", async () => {
      await startReceiver();
      const res = await get(port, "/v1/traces");
      expect(res.status).toBe(405);
    });

    it("returns 404 for unknown paths", async () => {
      await startReceiver();
      const res = await post(port, "/v1/unknown", {});
      expect(res.status).toBe(404);
    });
  });

  describe("POST /v1/register", () => {
    it("registers an app and returns its ID", async () => {
      await startReceiver();
      const res = await post(port, "/v1/register", {
        app_name: "my-app",
        app_version: "v1",
      });
      expect(res.status).toBe(200);
      const data = JSON.parse(res.body);
      expect(data.app_id).toMatch(/^app_hash_/);
      expect(stub.apps).toHaveLength(1);
      expect(stub.apps[0]!.appName).toBe("my-app");
      expect(stub.apps[0]!.appVersion).toBe("v1");
    });

    it("defaults app_version to 'base' when omitted", async () => {
      await startReceiver();
      await post(port, "/v1/register", { app_name: "default-ver" });
      expect(stub.apps[0]!.appVersion).toBe("base");
    });

    it("populates appJson with dashboard-compatible fields", async () => {
      await startReceiver();
      await post(port, "/v1/register", {
        app_name: "my-app",
        app_version: "v1",
      });
      const appJson = stub.apps[0]!.appJson;
      expect(appJson).toHaveProperty("metadata");
      expect(appJson).toHaveProperty("tags");
      expect(appJson).toHaveProperty("record_ingest_mode");
    });
  });

  describe("POST /v1/traces", () => {
    it("ingests spans and converts them to events", async () => {
      await startReceiver();

      const payload = otlpPayload(
        [
          {
            name: "test-span",
            traceId: "abcd1234",
            spanId: "span001",
            startTimeUnixNano: "1700000000000000000",
            endTimeUnixNano: "1700000001000000000",
            attributes: [
              { key: "ai.observability.span_type", value: strVal("tool") },
            ],
            status: { code: 1 },
          },
        ],
        [
          { key: "ai.observability.app_name", value: strVal("test-app") },
          { key: "ai.observability.app_version", value: strVal("v1") },
        ]
      );

      const res = await post(port, "/v1/traces", payload);
      expect(res.status).toBe(200);
      expect(stub.events).toHaveLength(1);

      const event = stub.events[0]!;
      expect(event.eventId).toBe("span001");
      expect(event.recordType).toBe("SPAN");
      expect(event.recordAttributes["ai.observability.span_type"]).toBe("tool");
      expect(event.resourceAttributes["ai.observability.app_name"]).toBe(
        "test-app"
      );
    });

    it("skips spans that lack an app_name resource attribute", async () => {
      await startReceiver();

      const payload = otlpPayload(
        [
          {
            name: "orphan-span",
            spanId: "span-orphan",
            attributes: [],
          },
        ],
        [] // no resource attributes
      );

      const res = await post(port, "/v1/traces", payload);
      expect(res.status).toBe(200);
      expect(stub.events).toHaveLength(0);
    });

    it("handles intValue (string representation)", async () => {
      await startReceiver();

      const payload = otlpPayload(
        [
          {
            name: "int-span",
            spanId: "span-int",
            attributes: [
              { key: "ai.observability.cost.num_tokens", value: intVal("42") },
            ],
          },
        ],
        [{ key: "ai.observability.app_name", value: strVal("app") }]
      );

      await post(port, "/v1/traces", payload);
      expect(stub.events).toHaveLength(1);
      expect(
        stub.events[0]!.recordAttributes["ai.observability.cost.num_tokens"]
      ).toBe(42);
    });

    it("handles arrayValue attributes", async () => {
      await startReceiver();

      const payload = otlpPayload(
        [
          {
            name: "array-span",
            spanId: "span-arr",
            attributes: [
              {
                key: "ai.observability.retrieval.retrieved_contexts",
                value: {
                  arrayValue: {
                    values: [strVal("ctx1"), strVal("ctx2")],
                  },
                },
              },
            ],
          },
        ],
        [{ key: "ai.observability.app_name", value: strVal("app") }]
      );

      await post(port, "/v1/traces", payload);
      expect(stub.events).toHaveLength(1);
      expect(
        stub.events[0]!.recordAttributes[
          "ai.observability.retrieval.retrieved_contexts"
        ]
      ).toEqual(["ctx1", "ctx2"]);
    });

    it("converts nano timestamps to Date objects", async () => {
      await startReceiver();

      const nanos = "1700000000000000000"; // 2023-11-14T22:13:20.000Z
      const payload = otlpPayload(
        [
          {
            name: "ts-span",
            spanId: "span-ts",
            startTimeUnixNano: nanos,
            endTimeUnixNano: nanos,
            attributes: [],
          },
        ],
        [{ key: "ai.observability.app_name", value: strVal("app") }]
      );

      await post(port, "/v1/traces", payload);
      const event = stub.events[0]!;
      expect(event.startTimestamp).toBeInstanceOf(Date);
      expect(event.startTimestamp.toISOString()).toBe(
        "2023-11-14T22:13:20.000Z"
      );
    });

    it("populates trace fields (span_id, trace_id, parent_id)", async () => {
      await startReceiver();

      const payload = otlpPayload(
        [
          {
            name: "child",
            traceId: "trace-abc",
            spanId: "span-child",
            parentSpanId: "span-parent",
            attributes: [],
          },
        ],
        [{ key: "ai.observability.app_name", value: strVal("app") }]
      );

      await post(port, "/v1/traces", payload);
      const t = stub.events[0]!.trace;
      expect(t.span_id).toBe("span-child");
      expect(t.trace_id).toBe("trace-abc");
      expect(t.parent_id).toBe("span-parent");
    });

    it("sets status to STATUS_CODE_ERROR for code 2", async () => {
      await startReceiver();

      const payload = otlpPayload(
        [
          {
            name: "err-span",
            spanId: "span-err",
            attributes: [],
            status: { code: 2, message: "bad" },
          },
        ],
        [{ key: "ai.observability.app_name", value: strVal("app") }]
      );

      await post(port, "/v1/traces", payload);
      expect(stub.events[0]!.record.status).toBe("STATUS_CODE_ERROR");
    });
  });

  describe("stop()", () => {
    it("resolves immediately if server was never started", async () => {
      const r = new TruLensReceiver({
        connector: new StubConnector(),
        port: 0,
      });
      await expect(r.stop()).resolves.toBeUndefined();
    });
  });
});
