import { describe, expect, it } from 'vitest';
import { uniq } from './uniq';

describe(uniq.name, () => {
  it('should return an empty array when given an empty array', () => {
    const input: number[] = [];
    const result = uniq(input);
    expect(result).toEqual([]);
    expect(result).not.toBe(input); // New array reference
  });

  it('should return array with unique numbers', () => {
    const input = [1, 2, 2, 3, 3, 3, 4];
    const result = uniq(input);
    expect(result).toEqual([1, 2, 3, 4]);
  });

  it('should return array with unique strings', () => {
    const input = ['a', 'b', 'b', 'c', 'c', 'c'];
    const result = uniq(input);
    expect(result).toEqual(['a', 'b', 'c']);
  });

  it('should preserve the order of items', () => {
    const input = ['c', 'a', 'b', 'a', 'c'];
    const result = uniq(input);
    expect(result).toEqual(['c', 'a', 'b']);
  });

  it('should work with objects based on reference', () => {
    const obj1 = { id: 1 };
    const obj2 = { id: 2 };
    const obj3 = { id: 1 }; // Same data but different reference

    const input = [obj1, obj2, obj1, obj3];
    const result = uniq(input);

    expect(result).toEqual([obj1, obj2, obj3]);
    expect(result.length).toBe(3);
  });
});
