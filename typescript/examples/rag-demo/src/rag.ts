/**
 * SimpleRAG — a minimal retrieval-augmented generation pipeline.
 *
 * Uses a static in-memory document store (no external vector DB needed) so
 * the demo has zero extra dependencies beyond the OpenAI client.
 *
 * retrieve() is decorated with @instrumentDecorator to emit a RETRIEVAL span.
 * generate() is left uninstrumented — the OpenAI auto-instrumentation captures
 * the GENERATION span with token/cost data automatically.
 */

import OpenAI from "openai";
import { instrumentDecorator } from "@trulens/core";
import { SpanAttributes, SpanType } from "@trulens/semconv";

// ---------------------------------------------------------------------------
// Document store (static for demo purposes)
// ---------------------------------------------------------------------------

const DOCUMENTS = [
  "TruLens is an open-source library for evaluating and tracking LLM applications. It provides feedback functions to measure quality metrics like relevance, groundedness, and coherence.",
  "Retrieval-Augmented Generation (RAG) combines a retrieval step with an LLM generation step. The retriever fetches relevant context from a knowledge base, which is then passed to the LLM as additional grounding.",
  "OpenTelemetry (OTEL) is a vendor-neutral observability framework for distributed systems. It provides APIs and SDKs for generating and collecting traces, metrics, and logs.",
  "A feedback function in TruLens takes one or more spans from a trace and returns a score between 0 and 1 indicating the quality of that aspect of the LLM app.",
  "TruLens uses OTEL semantic conventions to tag spans with types like RETRIEVAL, GENERATION, RECORD_ROOT, and MCP. This allows the dashboard to understand the structure of a trace.",
];

// ---------------------------------------------------------------------------
// Instrumented RAG class
// ---------------------------------------------------------------------------

export class SimpleRAG {
  private readonly openai: OpenAI;

  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env["OPENAI_API_KEY"],
    });
  }

  /**
   * Retrieve the top-k documents whose text contains at least one token from
   * the query (naive keyword overlap — no embeddings needed for the demo).
   */
  @instrumentDecorator<[string, number], Promise<string[]>>({
    spanType: SpanType.RETRIEVAL,
    attributes: (ret, _err, query) => ({
      [SpanAttributes.RETRIEVAL.QUERY_TEXT]: query as string,
      [SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS]: JSON.stringify(ret),
      [SpanAttributes.RETRIEVAL.NUM_CONTEXTS]:
        (ret as string[] | undefined)?.length ?? 0,
    }),
  })
  async retrieve(query: string, topK: number = 2): Promise<string[]> {
    const queryTokens = new Set(
      query
        .toLowerCase()
        .split(/\W+/)
        .filter((t) => t.length > 3)
    );

    const scored = DOCUMENTS.map((doc) => {
      const docTokens = doc.toLowerCase().split(/\W+/);
      const overlap = docTokens.filter((t) => queryTokens.has(t)).length;
      return { doc, overlap };
    })
      .filter(({ overlap }) => overlap > 0)
      .sort((a, b) => b.overlap - a.overlap)
      .slice(0, topK)
      .map(({ doc }) => doc);

    return scored.length > 0 ? scored : [DOCUMENTS[0]!];
  }

  /**
   * Generate an answer from the LLM given a query and retrieved context.
   * No manual instrumentation needed — OpenAI auto-instrumentation captures
   * the GENERATION span with model, token counts, and cost.
   */
  async generate(query: string, contexts: string[]): Promise<string> {
    const contextText = contexts
      .map((c, i) => `[${i + 1}] ${c}`)
      .join("\n\n");

    const response = await this.openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content:
            "You are a helpful assistant. Answer the question using only " +
            "the provided context. Be concise.\n\nContext:\n" +
            contextText,
        },
        { role: "user", content: query },
      ],
      max_tokens: 256,
    });

    return response.choices[0]?.message.content ?? "";
  }

  /**
   * Full RAG pipeline: retrieve relevant docs, then generate an answer.
   * The RECORD_ROOT span is created automatically by TruApp.
   */
  async query(question: string): Promise<string> {
    const contexts = await this.retrieve(question);
    return this.generate(question, contexts);
  }
}
