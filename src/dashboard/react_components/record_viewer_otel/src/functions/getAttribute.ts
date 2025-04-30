import type { Attributes } from '../types/attributes';

export const getStringAttribute = (attributes: Attributes, key: string) => {
  const value = attributes[key];
  if (typeof value === 'string') return value;
  return null;
};

export const getNumericalAttribute = (attributes: Attributes, key: string) => {
  const value = attributes[key];
  if (typeof value === 'number') return value;
  return null;
};
