import { describe, expect, it } from 'vitest';
import { getSpanTypeTitle } from './getSpanTypeTitle';

describe(getSpanTypeTitle.name, () => {
  it('should return "Unknown" when spanType is empty', () => {
    expect(getSpanTypeTitle('')).toBe('Unknown');
  });

  it('should return "Unknown" when spanType is undefined', () => {
    expect(getSpanTypeTitle(undefined as unknown as string)).toBe('Unknown');
  });

  it('should capitalize the first letter of the first word', () => {
    expect(getSpanTypeTitle('reranking')).toBe('Reranking');
  });

  it('should replace underscores with spaces', () => {
    expect(getSpanTypeTitle('record_root')).toBe('Record root');
  });
});
