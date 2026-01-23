import { splitStringToTwoPartsByDelimiter } from './splitStringToTwoPartsByDelimiter';
import { describe, expect, it } from 'vitest';

describe(splitStringToTwoPartsByDelimiter.name, () => {
  it('should split string correctly with single delimiter', () => {
    const input = 'a,b';
    const delimiter = ',';
    const result = splitStringToTwoPartsByDelimiter(input, delimiter);

    expect(result).toEqual([['a', 'b']]);
  });

  it('should split string correctly with multiple delimiters', () => {
    const input = 'a,b,c';
    const delimiter = ',';
    const result = splitStringToTwoPartsByDelimiter(input, delimiter);

    expect(result).toEqual([
      ['a,b', 'c'],
      ['a', 'b,c'],
    ]);
  });

  it('should return empty array if no delimiter is found', () => {
    const input = 'abc';
    const delimiter = ',';
    const result = splitStringToTwoPartsByDelimiter(input, delimiter);

    expect(result).toEqual([]);
  });

  it('should handle empty input string', () => {
    const input = '';
    const delimiter = ',';
    const result = splitStringToTwoPartsByDelimiter(input, delimiter);

    expect(result).toEqual([]);
  });

  it('should handle delimiter at the start and end of the string', () => {
    const input = ',a,b,';
    const delimiter = ',';
    const result = splitStringToTwoPartsByDelimiter(input, delimiter);

    expect(result).toEqual([
      [',a,b', ''],
      [',a', 'b,'],
      ['', 'a,b,'],
    ]);
  });
});
