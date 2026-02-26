import { afterEach, describe, expect, it } from "vitest";
import { SQLiteConnector } from "../sqlite-connector.js";
import type { EventRecord } from "../db-connector.js";

function makeConnector(): SQLiteConnector {
  return new SQLiteConnector({ dbPath: ":memory:" });
}

function makeEvent(overrides: Partial<EventRecord> = {}): EventRecord {
  return {
    eventId: overrides.eventId ?? "evt-1",
    record: overrides.record ?? { name: "test-span", kind: 1 },
    recordAttributes: overrides.recordAttributes ?? { "ai.observability.span_type": "tool" },
    recordType: "SPAN",
    resourceAttributes: overrides.resourceAttributes ?? { "ai.observability.app_name": "test" },
    startTimestamp: overrides.startTimestamp ?? new Date("2025-01-01T00:00:00Z"),
    timestamp: overrides.timestamp ?? new Date("2025-01-01T00:00:01Z"),
    trace: overrides.trace ?? { span_id: "abc", trace_id: "xyz", parent_id: "" },
  };
}

let connector: SQLiteConnector | null = null;

afterEach(() => {
  if (connector) {
    connector.close();
    connector = null;
  }
});

describe("SQLiteConnector", () => {
  describe("table creation", () => {
    it("creates all required tables on construction", () => {
      connector = makeConnector();
      // Access the internal db via a quick read query for each table
      const db = (connector as any).db;
      const tables = db
        .prepare(
          `SELECT name FROM sqlite_master WHERE type='table' ORDER BY name`
        )
        .all()
        .map((r: any) => r.name);

      expect(tables).toContain("trulens_apps");
      expect(tables).toContain("trulens_events");
      expect(tables).toContain("trulens_records");
      expect(tables).toContain("trulens_feedback_defs");
      expect(tables).toContain("trulens_feedbacks");
      expect(tables).toContain("trulens_alembic_version");
    });

    it("stamps alembic version as '10'", () => {
      connector = makeConnector();
      const db = (connector as any).db;
      const row = db
        .prepare("SELECT version_num FROM trulens_alembic_version LIMIT 1")
        .get() as { version_num: string };
      expect(row.version_num).toBe("10");
    });

    it("does not duplicate alembic version on re-open", () => {
      // Can't truly re-open :memory:, but we can call _createTables twice
      connector = makeConnector();
      const db = (connector as any).db;
      (connector as any)._createTables();
      const rows = db
        .prepare("SELECT version_num FROM trulens_alembic_version")
        .all();
      expect(rows).toHaveLength(1);
    });
  });

  describe("addApp()", () => {
    it("inserts a new app and returns the app ID", () => {
      connector = makeConnector();
      const id = connector.addApp({
        appId: "app_hash_abc123",
        appName: "test-app",
        appVersion: "v1",
        appJson: { app_id: "app_hash_abc123", app_name: "test-app" },
      });
      expect(id).toBe("app_hash_abc123");

      const db = (connector as any).db;
      const row = db
        .prepare("SELECT * FROM trulens_apps WHERE app_id = ?")
        .get("app_hash_abc123") as any;
      expect(row.app_name).toBe("test-app");
      expect(row.app_version).toBe("v1");
      expect(JSON.parse(row.app_json).app_id).toBe("app_hash_abc123");
    });

    it("upserts (replaces) on duplicate app_id", () => {
      connector = makeConnector();
      connector.addApp({
        appId: "app_hash_abc123",
        appName: "test-app",
        appVersion: "v1",
        appJson: { version: 1 },
      });
      connector.addApp({
        appId: "app_hash_abc123",
        appName: "test-app",
        appVersion: "v1",
        appJson: { version: 2 },
      });

      const db = (connector as any).db;
      const rows = db
        .prepare("SELECT * FROM trulens_apps WHERE app_id = ?")
        .all("app_hash_abc123");
      expect(rows).toHaveLength(1);
      expect(JSON.parse((rows[0] as any).app_json).version).toBe(2);
    });
  });

  describe("addEvents()", () => {
    it("inserts events and returns event IDs", () => {
      connector = makeConnector();
      const ids = connector.addEvents([
        makeEvent({ eventId: "evt-1" }),
        makeEvent({ eventId: "evt-2" }),
      ]);
      expect(ids).toEqual(["evt-1", "evt-2"]);

      const db = (connector as any).db;
      const count = db
        .prepare("SELECT COUNT(*) as cnt FROM trulens_events")
        .get() as any;
      expect(count.cnt).toBe(2);
    });

    it("stores serialised JSON for record, attributes, and trace", () => {
      connector = makeConnector();
      const event = makeEvent({
        recordAttributes: { "ai.observability.span_type": "retrieval" },
      });
      connector.addEvents([event]);

      const db = (connector as any).db;
      const row = db
        .prepare("SELECT * FROM trulens_events WHERE event_id = ?")
        .get("evt-1") as any;
      expect(JSON.parse(row.record_attributes)).toEqual({
        "ai.observability.span_type": "retrieval",
      });
    });

    it("stores ISO timestamps", () => {
      connector = makeConnector();
      const start = new Date("2025-06-15T12:00:00Z");
      const end = new Date("2025-06-15T12:00:01Z");
      connector.addEvents([
        makeEvent({ startTimestamp: start, timestamp: end }),
      ]);

      const db = (connector as any).db;
      const row = db
        .prepare("SELECT * FROM trulens_events WHERE event_id = ?")
        .get("evt-1") as any;
      expect(row.start_timestamp).toBe(start.toISOString());
      expect(row.timestamp).toBe(end.toISOString());
    });

    it("upserts on duplicate event_id", () => {
      connector = makeConnector();
      connector.addEvents([
        makeEvent({ eventId: "evt-dup", record: { name: "first" } }),
      ]);
      connector.addEvents([
        makeEvent({ eventId: "evt-dup", record: { name: "second" } }),
      ]);

      const db = (connector as any).db;
      const rows = db
        .prepare("SELECT * FROM trulens_events WHERE event_id = ?")
        .all("evt-dup");
      expect(rows).toHaveLength(1);
      expect(JSON.parse((rows[0] as any).record).name).toBe("second");
    });

    it("handles empty batch", () => {
      connector = makeConnector();
      const ids = connector.addEvents([]);
      expect(ids).toEqual([]);
    });
  });

  describe("close()", () => {
    it("does not throw on close", () => {
      connector = makeConnector();
      expect(() => connector!.close()).not.toThrow();
      connector = null; // prevent double-close in afterEach
    });
  });

  describe("custom table prefix", () => {
    it("uses a custom prefix for all tables", () => {
      connector = new SQLiteConnector({
        dbPath: ":memory:",
        tablePrefix: "custom_",
      });
      const db = (connector as any).db;
      const tables = db
        .prepare(
          `SELECT name FROM sqlite_master WHERE type='table' ORDER BY name`
        )
        .all()
        .map((r: any) => r.name);

      expect(tables).toContain("custom_apps");
      expect(tables).toContain("custom_events");
      expect(tables).toContain("custom_records");
    });
  });
});
