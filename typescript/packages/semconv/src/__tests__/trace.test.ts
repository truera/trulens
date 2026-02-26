import { describe, expect, it } from "vitest";

import { ResourceAttributes, SpanAttributes, SpanType } from "../trace.js";

describe("ResourceAttributes", () => {
  it("has the correct string values", () => {
    expect(ResourceAttributes.APP_ID).toBe("ai.observability.app_id");
    expect(ResourceAttributes.APP_NAME).toBe("ai.observability.app_name");
    expect(ResourceAttributes.APP_VERSION).toBe(
      "ai.observability.app_version"
    );
  });
});

describe("SpanType", () => {
  it("has RETRIEVAL = 'retrieval'", () => {
    expect(SpanType.RETRIEVAL).toBe("retrieval");
  });

  it("has RECORD_ROOT = 'record_root'", () => {
    expect(SpanType.RECORD_ROOT).toBe("record_root");
  });

  it("has MCP = 'MCP' (uppercase, mirrors Python)", () => {
    expect(SpanType.MCP).toBe("MCP");
  });

  it("has RERANKER = 'reranking' (matches Python)", () => {
    expect(SpanType.RERANKER).toBe("reranking");
  });
});

describe("SpanAttributes", () => {
  it("top-level keys use the ai.observability base scope", () => {
    expect(SpanAttributes.SPAN_TYPE).toBe("ai.observability.span_type");
    expect(SpanAttributes.RECORD_ID).toBe("ai.observability.record_id");
    expect(SpanAttributes.RUN_NAME).toBe("ai.observability.run.name");
  });

  it("RETRIEVAL attributes are correct", () => {
    expect(SpanAttributes.RETRIEVAL.QUERY_TEXT).toBe(
      "ai.observability.retrieval.query_text"
    );
    expect(SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS).toBe(
      "ai.observability.retrieval.retrieved_contexts"
    );
    expect(SpanAttributes.RETRIEVAL.NUM_CONTEXTS).toBe(
      "ai.observability.retrieval.num_contexts"
    );
  });

  it("RECORD_ROOT attributes are correct", () => {
    expect(SpanAttributes.RECORD_ROOT.INPUT).toBe(
      "ai.observability.record_root.input"
    );
    expect(SpanAttributes.RECORD_ROOT.OUTPUT).toBe(
      "ai.observability.record_root.output"
    );
    expect(SpanAttributes.RECORD_ROOT.ERROR).toBe(
      "ai.observability.record_root.error"
    );
  });

  it("MCP attributes are correct", () => {
    expect(SpanAttributes.MCP.TOOL_NAME).toBe(
      "ai.observability.mcp.tool_name"
    );
    expect(SpanAttributes.MCP.SERVER_NAME).toBe(
      "ai.observability.mcp.server_name"
    );
  });

  it("COST attributes are correct", () => {
    expect(SpanAttributes.COST.NUM_TOKENS).toBe(
      "ai.observability.cost.num_tokens"
    );
  });

  it("all attribute values match their Python equivalents", () => {
    // Spot-check a cross-section to catch regressions if trace.py changes.
    const expected: Record<string, string> = {
      [SpanAttributes.EVAL_ROOT.SCORE]: "ai.observability.eval_root.score",
      [SpanAttributes.RERANKER.OUTPUT_RANKS]:
        "ai.observability.reranking.output_ranks",
      [SpanAttributes.GRAPH_NODE.NODE_NAME]:
        "ai.observability.graph_node.node_name",
      [SpanAttributes.CALL.FUNCTION]: "ai.observability.call.function",
    };
    for (const [k, v] of Object.entries(expected)) {
      expect(k).toBe(v);
    }
  });
});
