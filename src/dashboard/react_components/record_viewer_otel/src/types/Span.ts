export interface Span {
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
