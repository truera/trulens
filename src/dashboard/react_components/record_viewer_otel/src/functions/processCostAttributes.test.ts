import { SpanAttributes } from '@/constants/span';
import { describe, it, expect } from 'vitest';
import { processCostAttributes } from './processCostAttributes';

/**
 * @jest-environment jsdom
 */

describe(processCostAttributes.name, () => {
  it('should return early if cost value is not present', () => {
    const attributes = {
      [SpanAttributes.COST_COST_CURRENCY]: 'USD',
    };

    processCostAttributes(attributes);

    expect(attributes).toEqual({
      [SpanAttributes.COST_COST_CURRENCY]: 'USD',
    });
  });

  it('should return early if cost currency is not present', () => {
    const attributes = {
      [SpanAttributes.COST_COST]: 123.45,
    };

    processCostAttributes(attributes);

    expect(attributes).toEqual({
      [SpanAttributes.COST_COST]: 123.45,
    });
  });

  it('should format recognized currency correctly', () => {
    const attributes = {
      [SpanAttributes.COST_COST]: 123.45,
      [SpanAttributes.COST_COST_CURRENCY]: 'USD',
    };

    processCostAttributes(attributes);

    expect(attributes).toEqual({
      [SpanAttributes.COST_COST]: '$123.45',
    });
  });

  it('should format small values with more decimal places', () => {
    const attributes = {
      [SpanAttributes.COST_COST]: 0.000123,
      [SpanAttributes.COST_COST_CURRENCY]: 'Snowflake Credits',
    };

    processCostAttributes(attributes);

    expect(attributes).toEqual({
      [SpanAttributes.COST_COST]: '0.00012 Snowflake Credits',
    });
  });
});
