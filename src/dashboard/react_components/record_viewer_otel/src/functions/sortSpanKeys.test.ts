import { SpanAttributes } from '@/constants/span';
import { describe, expect, it } from 'vitest';
import { sortSpanKeys } from './sortSpanKeys';

describe(sortSpanKeys.name, () => {
  it('should sort keys based on locale comparison of span attribute names', () => {
    const keyA = SpanAttributes.RECORD_ID;
    const keyB = SpanAttributes.APP_NAME;

    const result = sortSpanKeys(keyA, keyB);

    expect(result).toBe(1);
  });

  it('should return 0 when span attribute names are equal', () => {
    const keyA = SpanAttributes.RECORD_ID;
    const keyB = SpanAttributes.RECORD_ID;

    const result = sortSpanKeys(keyA, keyB);

    expect(result).toBe(0);
  });

  it('should return a negative number when first key is smaller', () => {
    const keyA = SpanAttributes.APP_NAME;
    const keyB = SpanAttributes.RECORD_ID;

    const result = sortSpanKeys(keyA, keyB);

    expect(result).toBe(-1);
  });
});
