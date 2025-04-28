export interface StackJSONRaw {
  event_id: string;
  record: {
    name: string;
    parent_span_id: string;
    status: string;
  };
  record_attributes: Record<string, unknown>;
  start_timestamp: number;
  timestamp: number;
  trace: {
    trace_id: string;
    parent_id: string;
    span_id: string;
  };
}

export type Span = StackJSONRaw;

export type CallJSONRaw = Record<string, any>;

export interface RecordJSONRaw {
  record_id: string;
  app_id: string;
  cost: {
    n_requests: number;
    n_successful_requests: number;
    n_classes: number;
    n_tokens: number;
    n_prompt_tokens: number;
    n_completion_tokens: number;
    cost: number;
  };
  ts: string;
  tags: string;
  main_input: string;
  main_output: string;
  main_error: string;
  calls: CallJSONRaw[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [others: string]: any;
}

export interface DataRaw {
  spans: Span[];
}
