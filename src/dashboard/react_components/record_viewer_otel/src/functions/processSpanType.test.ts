import { SpanAttributes, SpanType } from '@/constants/span';
import { processSpanType } from './processSpanType';
import { describe, it, expect } from 'vitest';

describe(processSpanType.name, () => {
  it('should return early if SPAN_TYPE attribute is not present', () => {
    const attributes = {};

    processSpanType(attributes);

    expect(attributes).toEqual({});
  });

  it('should replace UNKNOWN span type with "Not Specified"', () => {
    const attributes = {
      [SpanAttributes.SPAN_TYPE]: SpanType.UNKNOWN,
    };

    processSpanType(attributes);

    expect(attributes[SpanAttributes.SPAN_TYPE]).toBe('Not Specified');
  });

  it('should keep non-UNKNOWN span type as is', () => {
    const attributes = {
      [SpanAttributes.SPAN_TYPE]: 'record_root',
    };

    processSpanType(attributes);

    expect(attributes[SpanAttributes.SPAN_TYPE]).toBe('record_root');
  });
});
