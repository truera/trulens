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
  numContexts: number;

  // The retrieved contexts.
  retrievedContexts: string[] | null;

  constructor(rawSpan: SpanRaw) {
    super(rawSpan);
    this.type = SpanType.RETRIEVER;

    this.inputText = (this.getAttribute('input_text') as string) ?? null;
    this.inputEmbedding = (this.getAttribute('input_embedding') as number[]) ?? null;
    this.distanceType = (this.getAttribute('distance_type') as string) ?? null;
    this.numContexts = (this.getAttribute('num_contexts') as number) ?? null;
    this.retrievedContexts = (this.getAttribute('retrieved_contexts') as string[]) ?? null;
  }
}

export class SpanReranker extends Span {
  constructor(rawSpan: SpanRaw) {
    super(rawSpan);
    this.type = SpanType.RERANKER;
  }
}

export class SpanLLM extends Span {
  // The model name of the LLM
  modelName: string | null;

  constructor(rawSpan: SpanRaw) {
    super(rawSpan);
    this.type = SpanType.LLM;
    this.modelName = (this.getAttribute('model_name') as string) ?? null;
  }
}

export class SpanEmbedding extends Span {
  constructor(rawSpan: SpanRaw) {
    super(rawSpan);
    this.type = SpanType.EMBEDDING;
  }
}

export class SpanTool extends Span {
  constructor(rawSpan: SpanRaw) {
    super(rawSpan);
    this.type = SpanType.TOOL;
  }
}

export class SpanAgent extends Span {
  constructor(rawSpan: SpanRaw) {
    super(rawSpan);
    this.type = SpanType.AGENT;
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
    default:
      return new Span(rawSpan);
  }
};
