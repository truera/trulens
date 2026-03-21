import { afterEach, describe, expect, it } from "vitest";
import { InMemorySpanExporter } from "@opentelemetry/sdk-trace-node";

import { TruSession } from "../session.js";
import { SQLiteConnector } from "../sqlite-connector.js";

afterEach(async () => {
  // Ensure the singleton is torn down between tests.
  try {
    const s = TruSession.getInstance();
    await s.shutdown();
  } catch {
    // not initialised — fine
  }
});

describe("TruSession", () => {
  describe("validation", () => {
    it("throws when both connector and exporter are provided", async () => {
      await expect(
        TruSession.init({
          appName: "test",
          appVersion: "v1",
          connector: new SQLiteConnector({ dbPath: ":memory:" }),
          exporter: new InMemorySpanExporter(),
        })
      ).rejects.toThrow("mutually exclusive");
    });

    it("throws when neither connector nor exporter is provided", async () => {
      await expect(
        TruSession.init({
          appName: "test",
          appVersion: "v1",
        })
      ).rejects.toThrow("either");
    });
  });

  describe("exporter mode", () => {
    it("initialises with a custom exporter", async () => {
      const exporter = new InMemorySpanExporter();
      const session = await TruSession.init({
        appName: "exporter-app",
        appVersion: "v1",
        exporter,
      });

      expect(session.appName).toBe("exporter-app");
      expect(session.appVersion).toBe("v1");
      expect(session.appId).toMatch(/^app_hash_/);
    });

    it("exposes the same instance via getInstance()", async () => {
      const exporter = new InMemorySpanExporter();
      const session = await TruSession.init({
        appName: "singleton-test",
        appVersion: "v1",
        exporter,
      });

      expect(TruSession.getInstance()).toBe(session);
    });
  });

  describe("singleton behaviour", () => {
    it("throws from getInstance() before init()", () => {
      expect(() => TruSession.getInstance()).toThrow("not been initialised");
    });

    it("shuts down the previous session on re-init", async () => {
      const exp1 = new InMemorySpanExporter();
      const s1 = await TruSession.init({
        appName: "first",
        appVersion: "v1",
        exporter: exp1,
      });

      const exp2 = new InMemorySpanExporter();
      const s2 = await TruSession.init({
        appName: "second",
        appVersion: "v1",
        exporter: exp2,
      });

      expect(s2).not.toBe(s1);
      expect(TruSession.getInstance()).toBe(s2);
    });
  });

  describe("shutdown()", () => {
    it("clears the singleton after shutdown", async () => {
      const exporter = new InMemorySpanExporter();
      const session = await TruSession.init({
        appName: "shutdown-test",
        appVersion: "v1",
        exporter,
      });

      await session.shutdown();
      expect(() => TruSession.getInstance()).toThrow("not been initialised");
    });

    it("invokes onShutdown callback with inputRecordsCount", async () => {
      let capturedCount = -1;
      const exporter = new InMemorySpanExporter();
      const session = await TruSession.init({
        appName: "hook-test",
        appVersion: "v1",
        exporter,
        onShutdown: async (count) => {
          capturedCount = count;
        },
      });

      session.inputRecordsCount = 5;
      await session.shutdown();
      expect(capturedCount).toBe(5);
    });
  });

  describe("connector mode", () => {
    it("initialises with a SQLiteConnector and starts a receiver", async () => {
      const connector = new SQLiteConnector({ dbPath: ":memory:" });
      const session = await TruSession.init({
        appName: "connector-app",
        appVersion: "v1",
        connector,
        receiverPort: 24318,
      });

      expect(session.appName).toBe("connector-app");

      // The connector should have the app registered
      const db = (connector as any).db;
      const row = db
        .prepare("SELECT app_name FROM trulens_apps LIMIT 1")
        .get() as any;
      expect(row.app_name).toBe("connector-app");

      await session.shutdown();
    });
  });

  describe("lifecycle hooks", () => {
    it("calls onInit before returning", async () => {
      let initCalled = false;
      const exporter = new InMemorySpanExporter();
      await TruSession.init({
        appName: "hook-app",
        appVersion: "v1",
        exporter,
        onInit: async () => {
          initCalled = true;
        },
      });

      expect(initCalled).toBe(true);
    });
  });

  describe("properties", () => {
    it("stores runName when provided", async () => {
      const exporter = new InMemorySpanExporter();
      const session = await TruSession.init({
        appName: "run-app",
        appVersion: "v1",
        exporter,
        runName: "my-run",
      });

      expect(session.runName).toBe("my-run");
    });

    it("accepts a custom appId", async () => {
      const exporter = new InMemorySpanExporter();
      const session = await TruSession.init({
        appName: "custom-id-app",
        appVersion: "v1",
        exporter,
        appId: "custom_id_12345",
      });

      expect(session.appId).toBe("custom_id_12345");
    });
  });
});
