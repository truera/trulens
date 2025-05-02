import { Span } from '../types/Span';

/**
 * Creates a Span with reasonable defaults for missing fields.
 *
 * @param options User-provided fields for the Span
 * @returns A new Span object
 */
export function createSpan(
  options: Partial<{
    event_id: string;
    record_name: string;
    record_parent_span_id: string;
    record_status: string;
    record_attributes: Record<string, unknown>;
    start_timestamp: number;
    timestamp: number;
    trace_id: string;
    parent_id: string;
    span_id: string;
  }> = {}
): Span {
  const now = Date.now();

  return {
    event_id: options.event_id ?? `event-${Math.random().toString(36).substring(2, 11)}`,
    record: {
      name: options.record_name ?? 'unnamed-span',
      parent_span_id: options.record_parent_span_id ?? '',
      status: options.record_status ?? 'OK',
    },
    record_attributes: options.record_attributes || {},
    start_timestamp: options.start_timestamp ?? now,
    timestamp: options.timestamp ?? now + 100, // Default 100ms duration
    trace: {
      trace_id: options.trace_id ?? `trace-${Math.random().toString(36).substring(2, 11)}`,
      parent_id: options.parent_id ?? '',
      span_id: options.span_id ?? `span-${Math.random().toString(36).substring(2, 11)}`,
    },
  };
}
