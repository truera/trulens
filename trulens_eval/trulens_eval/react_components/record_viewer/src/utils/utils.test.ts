// sum.test.js
import { describe, expect, test } from 'vitest';

import { formatDuration, formatTime, getMicroseconds } from '@/utils/utils';

describe('getMicroseconds', () => {
  test('getMicroseconds gives 0 with a null timestamp', () => {
    expect(getMicroseconds(null!)).toBe(0);
  });

  test('getMicroseconds gives 0 with an undefined timestamp', () => {
    expect(getMicroseconds(undefined!)).toBe(0);
  });

  test('getMicroseconds gives expected result with full timestamp', () => {
    expect(getMicroseconds('2024-06-10T12:27:07.701071')).toBe(1718047627701071);
  });

  test('getMicroseconds gives expected result with millisecond-level timestamp', () => {
    expect(getMicroseconds('2024-06-10T12:27:07.701')).toBe(1718047627701000);
  });
});

describe('formatDuration', () => {
  test('formatDuration gives empty string with a null duration', () => {
    expect(formatDuration(null!)).toBe('');
  });

  test('formatDuration gives empty string with an undefined duration', () => {
    expect(formatDuration(undefined!)).toBe('');
  });

  test('formatDuration returns microseconds if < 1000', () => {
    expect(formatDuration(23)).toBe('23 µs');
  });

  test('formatDuration returns milliseconds if < 1000000 and > 1000, rounding happens upwards', () => {
    expect(formatDuration(23500)).toBe('24 ms');
  });

  test('formatDuration returns milliseconds if < 1000000 and > 1000', () => {
    expect(formatDuration(23000)).toBe('23 ms');
  });

  test('formatDuration returns seconds if > 1000000', () => {
    expect(formatDuration(23000000)).toBe('23 s');
  });
});

describe('formatTime', () => {
  test('formatTime gives empty string with a null timestamp', () => {
    expect(formatTime(null!)).toBe('');
  });

  test('formatTime gives empty string with an undefined timestamp', () => {
    expect(formatTime(undefined!)).toBe('');
  });

  test('formatTime returns expected time string', () => {
    expect(formatTime(23)).toBe('12/31/1969 16:00:00.000023');
  });
});
