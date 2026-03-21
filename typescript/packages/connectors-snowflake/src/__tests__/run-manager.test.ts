import { describe, expect, it } from "vitest";

import { RunStatus, SnowflakeRunManager } from "../run-manager.js";

// ---------------------------------------------------------------------------
// Create a manager with a null connector — we only test the pure parsing
// methods that don't touch the database.
// ---------------------------------------------------------------------------

const manager = new SnowflakeRunManager({
  connector: null as any,
});

describe("SnowflakeRunManager._parseRunStatus()", () => {
  it("returns CANCELLED when run_status is CANCELLED", () => {
    expect(
      manager._parseRunStatus({ run_status: "CANCELLED" })
    ).toBe(RunStatus.CANCELLED);
  });

  it("returns CREATED when there are no invocations", () => {
    expect(
      manager._parseRunStatus({ run_metadata: {} })
    ).toBe(RunStatus.CREATED);
  });

  it("returns CREATED when run_metadata is missing", () => {
    expect(manager._parseRunStatus({})).toBe(RunStatus.CREATED);
  });

  it("returns CREATED when invocations object is empty", () => {
    expect(
      manager._parseRunStatus({
        run_metadata: { invocations: {} },
      })
    ).toBe(RunStatus.CREATED);
  });
});

describe("SnowflakeRunManager._latestInvocationStatus()", () => {
  it("returns INVOCATION_COMPLETED for COMPLETED status", () => {
    const invocations = {
      inv1: {
        start_time_ms: 1000,
        completion_status: { status: "COMPLETED" },
      },
    };
    expect(manager._latestInvocationStatus(invocations)).toBe(
      RunStatus.INVOCATION_COMPLETED
    );
  });

  it("returns INVOCATION_IN_PROGRESS for STARTED status", () => {
    const invocations = {
      inv1: {
        start_time_ms: 1000,
        completion_status: { status: "STARTED" },
      },
    };
    expect(manager._latestInvocationStatus(invocations)).toBe(
      RunStatus.INVOCATION_IN_PROGRESS
    );
  });

  it("returns INVOCATION_PARTIALLY_COMPLETED for PARTIALLY_COMPLETED", () => {
    const invocations = {
      inv1: {
        start_time_ms: 1000,
        completion_status: { status: "PARTIALLY_COMPLETED" },
      },
    };
    expect(manager._latestInvocationStatus(invocations)).toBe(
      RunStatus.INVOCATION_PARTIALLY_COMPLETED
    );
  });

  it("returns FAILED for FAILED status", () => {
    const invocations = {
      inv1: {
        start_time_ms: 1000,
        completion_status: { status: "FAILED" },
      },
    };
    expect(manager._latestInvocationStatus(invocations)).toBe(
      RunStatus.FAILED
    );
  });

  it("returns UNKNOWN for unrecognised status", () => {
    const invocations = {
      inv1: {
        start_time_ms: 1000,
        completion_status: { status: "SOMETHING_ELSE" },
      },
    };
    expect(manager._latestInvocationStatus(invocations)).toBe(
      RunStatus.UNKNOWN
    );
  });

  it("picks the latest invocation by start_time_ms", () => {
    const invocations = {
      inv_old: {
        start_time_ms: 100,
        completion_status: { status: "COMPLETED" },
      },
      inv_new: {
        start_time_ms: 200,
        completion_status: { status: "STARTED" },
      },
    };
    expect(manager._latestInvocationStatus(invocations)).toBe(
      RunStatus.INVOCATION_IN_PROGRESS
    );
  });
});

describe("SnowflakeRunManager._overallComputationStatus()", () => {
  const completedInvocations = {
    inv1: {
      start_time_ms: 1000,
      completion_status: { status: "COMPLETED" },
    },
  };

  const failedInvocations = {
    inv1: {
      start_time_ms: 1000,
      completion_status: { status: "FAILED" },
    },
  };

  it("returns COMPLETED when all metrics and invocations completed", () => {
    const metrics = {
      m1: { completion_status: { status: "COMPLETED" } },
      m2: { completion_status: { status: "COMPLETED" } },
    };
    expect(
      manager._overallComputationStatus(completedInvocations, metrics)
    ).toBe(RunStatus.COMPLETED);
  });

  it("returns PARTIALLY_COMPLETED when metrics done but invocation not", () => {
    const metrics = {
      m1: { completion_status: { status: "COMPLETED" } },
    };
    expect(
      manager._overallComputationStatus(failedInvocations, metrics)
    ).toBe(RunStatus.PARTIALLY_COMPLETED);
  });

  it("returns FAILED when all metrics failed", () => {
    const metrics = {
      m1: { completion_status: { status: "FAILED" } },
      m2: { completion_status: { status: "FAILED" } },
    };
    expect(
      manager._overallComputationStatus(completedInvocations, metrics)
    ).toBe(RunStatus.FAILED);
  });

  it("returns COMPUTATION_IN_PROGRESS when metrics are mixed", () => {
    const metrics = {
      m1: { completion_status: { status: "COMPLETED" } },
      m2: { completion_status: { status: "STARTED" } },
    };
    expect(
      manager._overallComputationStatus(completedInvocations, metrics)
    ).toBe(RunStatus.COMPUTATION_IN_PROGRESS);
  });
});

describe("SnowflakeRunManager._parseRunStatus() integration", () => {
  it("delegates to _latestInvocationStatus when no metrics", () => {
    const meta = {
      run_metadata: {
        invocations: {
          inv1: {
            start_time_ms: 1000,
            completion_status: { status: "COMPLETED" },
          },
        },
      },
    };
    expect(manager._parseRunStatus(meta)).toBe(
      RunStatus.INVOCATION_COMPLETED
    );
  });

  it("delegates to _overallComputationStatus when metrics present", () => {
    const meta = {
      run_metadata: {
        invocations: {
          inv1: {
            start_time_ms: 1000,
            completion_status: { status: "COMPLETED" },
          },
        },
        metrics: {
          m1: { completion_status: { status: "COMPLETED" } },
        },
      },
    };
    expect(manager._parseRunStatus(meta)).toBe(RunStatus.COMPLETED);
  });
});
