import type {
  Span as SpanRaw,
  SpanAgent as SpanAgentRaw,
  SpanEmbedding as SpanEmbeddingRaw,
  SpanLLM as SpanLLMRaw,
  SpanMemory as SpanMemoryRaw,
  SpanOther as SpanOtherRaw,
  SpanReranker as SpanRerankerRaw,
  SpanRetriever as SpanRetrieverRaw,
  SpanRoot as SpanRootRaw,
  SpanTask as SpanTaskRaw,
  SpanTool as SpanToolRaw,
  SpanUntyped as SpanUntypedRaw,
} from '@/schema/span';
import { StatusCode } from '@/schema/spanAugment';

export enum SpanType {
  UNTYPED = 'SpanUntyped',
  ROOT = 'SpanRoot',
  RETRIEVER = 'SpanRetriever',
  RERANKER = 'SpanReranker',
  LLM = 'SpanLLM',
  EMBEDDING = 'SpanEmbedding',
  TOOL = 'SpanTool',
  AGENT = 'SpanAgent',
  TASK = 'SpanTask',
  OTHER = 'SpanOther',
  MEMORY = 'SpanMemory',
}

/**
 *  Type guards
 */
const getSpanType = (rawSpan: SpanRaw) => rawSpan.attributes?.[Span.vendorAttr('span_type')];

export const isSpanUntypedRaw = (rawSpan: SpanRaw): rawSpan is SpanUntypedRaw => {
  return getSpanType(rawSpan) === SpanType.UNTYPED;
};
export const isSpanRootRaw = (rawSpan: SpanRaw): rawSpan is SpanRootRaw => {
  return getSpanType(rawSpan) === SpanType.ROOT;
};
export const isSpanRetrieverRaw = (rawSpan: SpanRaw): rawSpan is SpanRetrieverRaw => {
  return getSpanType(rawSpan) === SpanType.RETRIEVER;
};
export const isSpanRerankerRaw = (rawSpan: SpanRaw): rawSpan is SpanRerankerRaw => {
  return getSpanType(rawSpan) === SpanType.RERANKER;
};
export const isSpanLLMRaw = (rawSpan: SpanRaw): rawSpan is SpanLLMRaw => {
  return getSpanType(rawSpan) === SpanType.LLM;
};
export const isSpanEmbeddingRaw = (rawSpan: SpanRaw): rawSpan is SpanEmbeddingRaw => {
  return getSpanType(rawSpan) === SpanType.EMBEDDING;
};
export const isSpanToolRaw = (rawSpan: SpanRaw): rawSpan is SpanToolRaw => {
  return getSpanType(rawSpan) === SpanType.TOOL;
};
export const isSpanAgentRaw = (rawSpan: SpanRaw): rawSpan is SpanAgentRaw => {
  return getSpanType(rawSpan) === SpanType.AGENT;
};
export const isSpanTaskRaw = (rawSpan: SpanRaw): rawSpan is SpanTaskRaw => {
  return getSpanType(rawSpan) === SpanType.TASK;
};
export const isSpanOtherRaw = (rawSpan: SpanRaw): rawSpan is SpanOtherRaw => {
  return getSpanType(rawSpan) === SpanType.OTHER;
};
export const isSpanMemoryRaw = (rawSpan: SpanRaw): rawSpan is SpanMemoryRaw => {
  return getSpanType(rawSpan) === SpanType.MEMORY;
};

/**
 * Utility function to convert the span types above to make them more human-readable
 * by removing the substring 'Span'.
 */
export const toHumanSpanType = (spanType?: SpanType) => {
  if (!spanType) return 'Other';

  return spanType.split('Span').join('');
};

/**
 * Base class for spans.
 *
 * Note: keep in sync with `trulens_eval/trace/span.py`.
 */
export class Span {
  // Identifier for the span
  spanId: number;

  // Identifier for the trace this span belongs to.
  traceId: number;

  // Tags associated with the span.
  tags: string[] = [];

  // Type of span.
  type: SpanType = SpanType.UNTYPED;

  // Metadata of span.
  metadata: Record<string, unknown>;

  // Name of span.
  name: string;

  // Start timestamp of span.
  startTimestamp: number | null;

  // End timestamp of span. Optional until the span finishes.
  endTimestamp: number | null;

  // Status of span.
  status: StatusCode | undefined;

  // Description of status, if available.
  statusDescription: string | null;

  // Span attributes.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  attributes: Record<string, any>;

  static vendorAttr(attributeName: string) {
    return `trulens_eval@${attributeName}`;
  }

  constructor(rawSpan: SpanRaw) {
    this.name = rawSpan.name;
    this.startTimestamp = rawSpan.start_timestamp ?? null;
    this.endTimestamp = rawSpan.end_timestamp ?? null;
    this.status = rawSpan.status;
    this.statusDescription = rawSpan.status_description ?? null;

    const [spanId, traceId] = rawSpan.context ?? [-1, -1];
    this.spanId = spanId;
    this.traceId = traceId;

    this.attributes = rawSpan.attributes ?? {};
    this.metadata = rawSpan.attributes_metadata ?? {};
  }

  getAttribute(attributeName: string) {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-return
    return this.attributes[Span.vendorAttr(attributeName)];
  }
}

export class SpanRoot extends Span {
  constructor(rawSpan: SpanRaw) {
    super(rawSpan);
    this.type = SpanType.ROOT;
  }
}

export class SpanRetriever extends Span {
  // Input text whose related contexts are being retrieved.
  inputText: string | null;

  // Embedding of the input text.
  inputEmbedding: number[] | null;

  // Distance function used for ranking contexts.
  distanceType: string | null;

  // The number of contexts requested, not necessarily retrieved.
  numContexts: number | null;

  // The retrieved contexts.
  rawRetrievedContexts: string[] | null;

  // The scores of the retrieved contexts.
  rawRetrievedScores: number[] | null;

  // The embeddings of the retrieved contexts.
  rawRetrievedEmbeddings: number[][] | null;

  // Processed contexts that zips up the raw contexts, scores, and embeddings
  retrievedContexts: { context: string | null; score: number | null; embedding: number[] | null }[] | null;

  constructor(rawSpan: SpanRetrieverRaw) {
    super(rawSpan);
    this.type = SpanType.RETRIEVER;

    this.inputText = (this.getAttribute('query_text') as typeof rawSpan.query_text) ?? null;
    this.inputEmbedding = (this.getAttribute('query_embedding') as typeof rawSpan.query_embedding) ?? null;
    this.distanceType = (this.getAttribute('distance_type') as typeof rawSpan.distance_type) ?? null;
    this.numContexts = (this.getAttribute('num_contexts') as typeof rawSpan.num_contexts) ?? null;
    this.rawRetrievedContexts = (this.getAttribute('retrieved_contexts') as typeof rawSpan.retrieved_contexts) ?? null;
    this.rawRetrievedScores = (this.getAttribute('retrieved_scores') as typeof rawSpan.retrieved_scores) ?? null;
    this.rawRetrievedEmbeddings =
      (this.getAttribute('retrieved_embeddings') as typeof rawSpan.retrieved_embeddings) ?? null;

    this.retrievedContexts = this.rawRetrievedContexts?.map((context, index) => ({
      context,
      score: this.rawRetrievedScores?.[index] ?? null,
      embedding: this.rawRetrievedEmbeddings?.[index] ?? null,
    }));
  }
}

export class SpanReranker extends Span {
  // The query text.
  queryText: string | null;

  // The model name of the reranker.
  modelName: string | null;

  // The number of contexts to rerank.
  topN: number | null;

  // The contexts being reranked.
  inputContextTexts: string[] | null;

  // The scores of the input contexts.
  inputContextScores: number[] | null;

  // Reranked indexes into `inputContextTexts`.
  outputRanks: number[] | null;

  contexts:
    | {
        context: string | null;
        inputScore: number | null;
        outputRank: number | null;
      }[]
    | null;

  constructor(rawSpan: SpanRerankerRaw) {
    super(rawSpan);
    this.type = SpanType.RERANKER;

    this.queryText = (this.getAttribute('query_text') as typeof rawSpan.query_text) ?? null;
    this.modelName = (this.getAttribute('model_name') as typeof rawSpan.model_name) ?? null;
    this.topN = (this.getAttribute('top_n') as typeof rawSpan.top_n) ?? null;
    this.inputContextTexts = (this.getAttribute('input_context_texts') as typeof rawSpan.input_context_texts) ?? null;
    this.inputContextScores =
      (this.getAttribute('input_context_scores') as typeof rawSpan.input_context_scores) ?? null;
    this.outputRanks = (this.getAttribute('output_ranks') as typeof rawSpan.output_ranks) ?? null;

    this.contexts = this.inputContextTexts?.map((context, index) => ({
      context,
      inputScore: this.inputContextScores?.[index] ?? null,
      outputRank: this.outputRanks?.[index] ?? null,
    }));
  }
}

export class SpanLLM extends Span {
  // The model name of the LLM
  modelName: string | null;

  // The type of model used.
  modelType: string | null;

  // The temperature used for generation.
  temperature: number | null;

  // The prompt given to the LLM.
  inputMessages: Record<string, unknown>[] | null;

  // The number of tokens in the input.
  inputTokenCount: number | null;

  // The returned text.
  outputMessages: Record<string, unknown>[] | null;

  // The number of tokens in the output.
  outputTokenCount: number | null;

  // The cost of the generation.
  cost: number | null;

  constructor(rawSpan: SpanLLMRaw) {
    super(rawSpan);
    this.type = SpanType.LLM;

    this.modelName = (this.getAttribute('model_name') as typeof rawSpan.model_name) ?? null;
    this.modelType = (this.getAttribute('model_type') as typeof rawSpan.model_type) ?? null;
    this.temperature = (this.getAttribute('temperature') as typeof rawSpan.temperature) ?? null;
    this.inputMessages = (this.getAttribute('input_messages') as typeof rawSpan.input_messages) ?? null;
    this.inputTokenCount = (this.getAttribute('input_token_count') as typeof rawSpan.input_token_count) ?? null;
    this.outputMessages = (this.getAttribute('output_messages') as typeof rawSpan.output_messages) ?? null;
    this.outputTokenCount = (this.getAttribute('output_token_count') as typeof rawSpan.output_token_count) ?? null;
    this.cost = (this.getAttribute('cost') as typeof rawSpan.cost) ?? null;
  }
}

export class SpanMemory extends Span {
  // The type of memory.
  memoryType: string | null;

  // The text being integrated into the memory in this span.
  remembered: string | null;

  constructor(rawSpan: SpanMemoryRaw) {
    super(rawSpan);
    this.type = SpanType.MEMORY;

    this.memoryType = (this.getAttribute('memory_type') as typeof rawSpan.memory_type) ?? null;
    this.remembered = (this.getAttribute('remembered') as typeof rawSpan.remembered) ?? null;
  }
}

export class SpanEmbedding extends Span {
  // The text being embedded.
  inputText: string | null;

  // The model name of the embedding model.
  modelName: string | null;

  // The embedding of the input text.
  embedding: number[] | null;

  constructor(rawSpan: SpanEmbeddingRaw) {
    super(rawSpan);
    this.type = SpanType.EMBEDDING;

    this.inputText = (this.getAttribute('input_text') as typeof rawSpan.input_text) ?? null;
    this.modelName = (this.getAttribute('model_name') as typeof rawSpan.model_name) ?? null;
    this.embedding = (this.getAttribute('embedding') as typeof rawSpan.embedding) ?? null;
  }
}

export class SpanTool extends Span {
  // The description of the tool.
  description: string | null;

  constructor(rawSpan: SpanToolRaw) {
    super(rawSpan);
    this.type = SpanType.TOOL;

    this.description = (this.getAttribute('description') as typeof rawSpan.description) ?? null;
  }
}

export class SpanAgent extends Span {
  // The description of the tool.
  description: string | null;

  constructor(rawSpan: SpanAgentRaw) {
    super(rawSpan);
    this.type = SpanType.AGENT;

    this.description = (this.getAttribute('description') as typeof rawSpan.description) ?? null;
  }
}

export class SpanTask extends Span {
  constructor(rawSpan: SpanTaskRaw) {
    super(rawSpan);
    this.type = SpanType.TASK;
  }
}

export class SpanOther extends Span {
  constructor(rawSpan: SpanOtherRaw) {
    super(rawSpan);
    this.type = SpanType.OTHER;
  }
}

export const createSpan = (rawSpan: SpanRaw) => {
  if (isSpanRootRaw(rawSpan)) return new SpanRoot(rawSpan);
  if (isSpanRetrieverRaw(rawSpan)) return new SpanRetriever(rawSpan);
  if (isSpanRerankerRaw(rawSpan)) return new SpanReranker(rawSpan);
  if (isSpanLLMRaw(rawSpan)) return new SpanLLM(rawSpan);
  if (isSpanEmbeddingRaw(rawSpan)) return new SpanEmbedding(rawSpan);
  if (isSpanToolRaw(rawSpan)) return new SpanTask(rawSpan);
  if (isSpanAgentRaw(rawSpan)) return new SpanAgent(rawSpan);
  if (isSpanTaskRaw(rawSpan)) return new SpanTask(rawSpan);
  if (isSpanOtherRaw(rawSpan)) return new SpanOther(rawSpan);
  if (isSpanMemoryRaw(rawSpan)) return new SpanMemory(rawSpan);

  return new Span(rawSpan);
};
