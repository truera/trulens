import { SpanRaw } from './types';

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
  metadata: Record<string, object | string>;

  // Name of span.
  name: string;

  // Start timestamp of span.
  startTimestamp: number;

  // End timestamp of span. Optional until the span finishes.
  endTimestamp: number | null;

  // Status of span.
  status: 'UNSET' | 'OK' | 'Error';

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
    this.startTimestamp = rawSpan.start_timestamp;
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
  retrievedContexts: string[] | null;

  // The scores of the retrieved contexts.
  retrievedScores: number[] | null;

  // The embeddings of the retrieved contexts.
  retrievedEmbeddings: number[][] | null;

  constructor(rawSpan: SpanRaw) {
    super(rawSpan);
    this.type = SpanType.RETRIEVER;

    this.inputText = (this.getAttribute('input_text') as string) ?? null;
    this.inputEmbedding = (this.getAttribute('input_embedding') as number[]) ?? null;
    this.distanceType = (this.getAttribute('distance_type') as string) ?? null;
    this.numContexts = (this.getAttribute('num_contexts') as number) ?? null;
    this.retrievedContexts = (this.getAttribute('retrieved_contexts') as string[]) ?? null;
    this.retrievedScores = (this.getAttribute('retrieved_scores') as number[]) ?? null;
    this.retrievedEmbeddings = (this.getAttribute('retrieved_embeddings') as number[][]) ?? null;
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

  constructor(rawSpan: SpanRaw) {
    super(rawSpan);
    this.type = SpanType.RERANKER;
    this.queryText = (this.getAttribute('query_text') as string) ?? null;
    this.modelName = (this.getAttribute('model_name') as string) ?? null;
    this.topN = (this.getAttribute('top_n') as number) ?? null;
    this.inputContextTexts = (this.getAttribute('input_context_texts') as string[]) ?? null;
    this.inputContextScores = (this.getAttribute('input_score_scores') as number[]) ?? null;
    this.outputRanks = (this.getAttribute('output_ranks') as number[]) ?? null;
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
  inputMessages: Record<string, string>[] | null;

  // The number of tokens in the input.
  inputTokenCount: number | null;

  // The returned text.
  outputMessages: Record<string, string>[] | null;

  // The number of tokens in the output.
  outputTokenCount: number | null;

  // The cost of the generation.
  cost: number | null;

  constructor(rawSpan: SpanRaw) {
    super(rawSpan);
    this.type = SpanType.LLM;
    this.modelName = (this.getAttribute('model_name') as string) ?? null;
    this.modelType = (this.getAttribute('model_type') as string) ?? null;
    this.temperature = (this.getAttribute('temperature') as number) ?? null;
    this.inputMessages = (this.getAttribute('input_messages') as Record<string, string>[]) ?? null;
    this.inputTokenCount = (this.getAttribute('input_token_count') as number) ?? null;
    this.outputMessages = (this.getAttribute('output_messages') as Record<string, string>[]) ?? null;
    this.outputTokenCount = (this.getAttribute('output_token_count') as number) ?? null;
    this.cost = (this.getAttribute('cost') as number) ?? null;
  }
}

export class SpanMemory extends Span {
  // The type of memory.
  memoryType: string | null;

  // The text being integrated into the memory in this span.
  remembered: string | null;

  constructor(rawSpan: SpanRaw) {
    super(rawSpan);
    this.type = SpanType.MEMORY;
    this.memoryType = (this.getAttribute('memory_type') as string) ?? null;
    this.remembered = (this.getAttribute('remembered') as string) ?? null;
  }
}

export class SpanEmbedding extends Span {
  // The text being embedded.
  inputText: string | null;

  // The model name of the embedding model.
  modelName: string | null;

  // The embedding of the input text.
  embedding: number[] | null;

  constructor(rawSpan: SpanRaw) {
    super(rawSpan);
    this.type = SpanType.EMBEDDING;
    this.inputText = (this.getAttribute('input_text') as string) ?? null;
    this.modelName = (this.getAttribute('model_name') as string) ?? null;
    this.embedding = (this.getAttribute('embedding') as number[]) ?? null;
  }
}

export class SpanTool extends Span {
  // The description of the tool.
  description: string | null;

  constructor(rawSpan: SpanRaw) {
    super(rawSpan);
    this.type = SpanType.TOOL;
    this.description = (this.getAttribute('description') as string) ?? null;
  }
}

export class SpanAgent extends Span {
  // The description of the tool.
  description: string | null;

  constructor(rawSpan: SpanRaw) {
    super(rawSpan);
    this.type = SpanType.AGENT;
    this.description = (this.getAttribute('description') as string) ?? null;
  }
}

export class SpanTask extends Span {
  constructor(rawSpan: SpanRaw) {
    super(rawSpan);
    this.type = SpanType.TASK;
  }
}

export class SpanOther extends Span {
  constructor(rawSpan: SpanRaw) {
    super(rawSpan);
    this.type = SpanType.OTHER;
  }
}

export const createSpan = (rawSpan: SpanRaw) => {
  const rawSpanType = rawSpan.attributes?.[Span.vendorAttr('span_type')] ?? SpanType.UNTYPED;

  switch (rawSpanType as SpanType) {
    case SpanType.ROOT:
      return new SpanRoot(rawSpan);
    case SpanType.RETRIEVER:
      return new SpanRetriever(rawSpan);
    case SpanType.RERANKER:
      return new SpanReranker(rawSpan);
    case SpanType.LLM:
      return new SpanLLM(rawSpan);
    case SpanType.EMBEDDING:
      return new SpanEmbedding(rawSpan);
    case SpanType.TOOL:
      return new SpanTool(rawSpan);
    case SpanType.AGENT:
      return new SpanAgent(rawSpan);
    case SpanType.TASK:
      return new SpanTask(rawSpan);
    case SpanType.OTHER:
      return new SpanOther(rawSpan);
    case SpanType.MEMORY:
      return new SpanMemory(rawSpan);
    default:
      return new Span(rawSpan);
  }
};
