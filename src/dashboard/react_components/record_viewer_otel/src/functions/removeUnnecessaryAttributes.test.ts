import { describe, it, expect } from 'vitest';
import { removeUnnecessaryAttributes } from './removeUnnecessaryAttributes';
import { SpanAttributes } from '@/constants/span';

describe(removeUnnecessaryAttributes.name, () => {
  it('should remove the name attribute', () => {
    const attributes = {
      name: 'test-name',
      key1: 'value1',
    };

    removeUnnecessaryAttributes(attributes);

    expect(attributes).toEqual({
      key1: 'value1',
    });
  });

  it('should remove app name and version attributes', () => {
    const attributes = {
      [SpanAttributes.APP_NAME]: 'test-app',
      [SpanAttributes.APP_VERSION]: '1.0.0',
      key1: 'value1',
    };

    removeUnnecessaryAttributes(attributes);

    expect(attributes).toEqual({
      key1: 'value1',
    });
  });

  it('should remove record ID and run name attributes', () => {
    const attributes = {
      [SpanAttributes.RECORD_ID]: 'record-123',
      [SpanAttributes.RUN_NAME]: 'run-abc',
      key1: 'value1',
    };

    removeUnnecessaryAttributes(attributes);

    expect(attributes).toEqual({
      key1: 'value1',
    });
  });

  it('should remove span type attribute', () => {
    const attributes = {
      [SpanAttributes.SPAN_TYPE]: 'http',
      key1: 'value1',
    };

    removeUnnecessaryAttributes(attributes);

    expect(attributes).toEqual({
      key1: 'value1',
    });
  });

  it('should remove multiple unnecessary attributes at once', () => {
    const attributes = {
      name: 'test-name',
      [SpanAttributes.APP_NAME]: 'test-app',
      [SpanAttributes.APP_VERSION]: '1.0.0',
      [SpanAttributes.RECORD_ID]: 'record-123',
      [SpanAttributes.RUN_NAME]: 'run-abc',
      [SpanAttributes.SPAN_TYPE]: 'http',
      key1: 'value1',
      key2: 'value2',
    };

    removeUnnecessaryAttributes(attributes);

    expect(attributes).toEqual({
      key1: 'value1',
      key2: 'value2',
    });
  });

  it('should not modify attributes when no unnecessary attributes are present', () => {
    const attributes = {
      key1: 'value1',
      key2: 'value2',
    };

    removeUnnecessaryAttributes(attributes);

    expect(attributes).toEqual({
      key1: 'value1',
      key2: 'value2',
    });
  });
});
