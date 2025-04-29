import { describe, it, expect } from 'vitest';
import { deduplicateAttributes } from './deduplicateAttributes';

describe(deduplicateAttributes.name, () => {
  it('should deduplicate attributes with same values', () => {
    const attributes = {
      key1: 'value1',
      key2: 'value1',
      key3: 'value2',
    };

    deduplicateAttributes(attributes);

    expect(attributes).toEqual({
      'key1 | key2': 'value1',
      key3: 'value2',
    });
  });

  it('should handle complex values', () => {
    const attributes = {
      key1: { a: 1 },
      key2: { a: 1 },
      key3: { b: 2 },
    };

    deduplicateAttributes(attributes);

    expect(attributes).toEqual({
      'key1 | key2': { a: 1 },
      key3: { b: 2 },
    });
  });

  it('should handle array values', () => {
    const attributes = {
      key1: [1, 2],
      key2: [1, 2],
      key3: [3, 4],
    };

    deduplicateAttributes(attributes);

    expect(attributes).toEqual({
      'key1 | key2': [1, 2],
      key3: [3, 4],
    });
  });

  it('should not modify attributes without duplicates', () => {
    const attributes = {
      key1: 'value1',
      key2: 'value2',
      key3: 'value3',
    };

    deduplicateAttributes(attributes);

    expect(attributes).toEqual({
      key1: 'value1',
      key2: 'value2',
      key3: 'value3',
    });
  });
});
