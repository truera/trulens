import { afterAll, beforeAll, describe, expect, test, vi } from 'vitest';
import { formatDuration, formatTime } from '@/functions/formatters';

describe('formatters', () => {
  beforeAll(() => {
    vi.stubGlobal('navigator', {
      languages: ['en-US'],
    });
  });

  afterAll(() => vi.unstubAllGlobals());

  describe(formatDuration.name, () => {
    test('formatDuration gives empty string with a null duration', () => {
      expect(formatDuration(null!)).toBe('');
    });

    test('formatDuration gives empty string with an undefined duration', () => {
      expect(formatDuration(undefined!)).toBe('');
    });

    test('formatDuration returns microseconds if < 0.001 seconds', () => {
      expect(formatDuration(0.0005)).toBe('500 µs');
    });

    test('formatDuration returns milliseconds if >= 0.001 seconds and < 1 second', () => {
      expect(formatDuration(0.023)).toBe('23 ms');
    });

    test('formatDuration returns milliseconds with proper formatting', () => {
      expect(formatDuration(0.5678)).toBe('567.8 ms');
    });

    test('formatDuration returns seconds if >= 1 second', () => {
      expect(formatDuration(23)).toBe('23 s');
    });
  });

  describe(formatTime.name, () => {
    test('formatTime gives empty string with a null timestamp', () => {
      expect(formatTime(null!)).toBe('');
    });

    test('formatTime gives empty string with an undefined timestamp', () => {
      expect(formatTime(undefined!)).toBe('');
    });

    test('formatTime returns expected time string', () => {
      expect(formatTime(23)).toBe('Wednesday, 12/31/1969, 4:00:23.000 PM PST');
    });
  });
});
