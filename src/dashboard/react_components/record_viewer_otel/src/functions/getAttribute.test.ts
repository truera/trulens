import type { Attributes } from '@/types/attributes';
import { describe, expect, it } from 'vitest';
import { getNumericalAttribute, getStringAttribute } from './getAttribute';

describe(getStringAttribute.name, () => {
  it('should return string value when attribute exists and is string', () => {
    const attributes: Attributes = { key: 'value' };
    expect(getStringAttribute(attributes, 'key')).toBe('value');
  });

  it('should return null when attribute exists but is not string', () => {
    const attributes: Attributes = { key: 123 };
    expect(getStringAttribute(attributes, 'key')).toBeNull();
  });

  it('should return null when attribute does not exist', () => {
    const attributes: Attributes = {};
    expect(getStringAttribute(attributes, 'key')).toBeNull();
  });
});

describe(getNumericalAttribute.name, () => {
  it('should return number value when attribute exists and is number', () => {
    const attributes: Attributes = { key: 123 };
    expect(getNumericalAttribute(attributes, 'key')).toBe(123);
  });

  it('should return null when attribute exists but is not number', () => {
    const attributes: Attributes = { key: 'value' };
    expect(getNumericalAttribute(attributes, 'key')).toBeNull();
  });

  it('should return null when attribute does not exist', () => {
    const attributes: Attributes = {};
    expect(getNumericalAttribute(attributes, 'key')).toBeNull();
  });
});
