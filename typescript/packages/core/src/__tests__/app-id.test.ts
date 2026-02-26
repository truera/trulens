import { describe, expect, it } from "vitest";
import { computeAppId } from "../app-id.js";

describe("computeAppId()", () => {
  it("produces a deterministic hash for the same inputs", () => {
    const a = computeAppId("my-app", "v1");
    const b = computeAppId("my-app", "v1");
    expect(a).toBe(b);
  });

  it("returns 'app_hash_' prefix followed by a 32-char hex string", () => {
    const id = computeAppId("test", "v1");
    expect(id).toMatch(/^app_hash_[0-9a-f]{32}$/);
  });

  it("produces different hashes for different app names", () => {
    const a = computeAppId("app-a", "v1");
    const b = computeAppId("app-b", "v1");
    expect(a).not.toBe(b);
  });

  it("produces different hashes for different versions", () => {
    const a = computeAppId("my-app", "v1");
    const b = computeAppId("my-app", "v2");
    expect(a).not.toBe(b);
  });

  it("matches Python's AppDefinition._compute_app_id for known inputs", () => {
    expect(computeAppId("my-app", "v1")).toBe(
      "app_hash_97e4360de06af798c5b11719c6e2f975"
    );
    expect(computeAppId("trulens-rag-demo", "v1")).toBe(
      "app_hash_bc68bb077845909fc79e5b14b6044b0b"
    );
    expect(computeAppId("my-app", "v2")).toBe(
      "app_hash_7af53691f002a42c7c138893c21c0b99"
    );
  });

  it("handles empty strings", () => {
    const id = computeAppId("", "");
    expect(id).toBe("app_hash_b1635c948820a6bf68664f49f9b3f807");
  });
});
