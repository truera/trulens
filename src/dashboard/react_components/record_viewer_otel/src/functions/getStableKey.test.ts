import { getStableKey } from './getStableKey';
import { describe, it, expect } from 'vitest';

describe(getStableKey.name, () => {
  it('should sort and stringify arrays', () => {
    const input = [3, 1, 2];
    const result = getStableKey(input);
    expect(result).toBe('[1,2,3]');
  });

  it('should sort and stringify object entries', () => {
    const input = { c: 3, a: 1, b: 2 };
    const result = getStableKey(input);
    expect(result).toBe('[["a",1],["b",2],["c",3]]');
  });

  it('should convert primitives to string', () => {
    expect(getStableKey(123)).toBe('123');
    expect(getStableKey('test')).toBe('test');
    expect(getStableKey(true)).toBe('true');
    expect(getStableKey(null)).toBe('null');
    expect(getStableKey(undefined)).toBe('undefined');
  });

  it('should handle nested objects and arrays', () => {
    const input = { arr: [3, 1], obj: { b: 2, a: 1 } };
    const result = getStableKey(input);
    expect(result).toBe('[["arr",[3,1]],["obj",{"b":2,"a":1}]]');
  });
});
