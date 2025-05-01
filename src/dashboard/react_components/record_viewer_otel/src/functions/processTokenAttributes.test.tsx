import { render, screen } from '@testing-library/react';
import { processTokenAttributes } from './processTokenAttributes';
import { SpanAttributes } from '@/constants/span';
import { describe, expect, it } from 'vitest';
import { ReactNode } from 'react';

/**
 * @jest-environment jsdom
 */

describe(processTokenAttributes.name, () => {
  it('should return early if no valid token attributes are present', () => {
    const attributes = {};
    const results = {};

    processTokenAttributes(attributes, results);

    expect(results).toEqual({});
  });

  it('should process token attributes and add them to results', () => {
    const attributes = {
      [SpanAttributes.COST_NUM_PROMPT_TOKENS]: 100,
      [SpanAttributes.COST_NUM_COMPLETION_TOKENS]: 50,
      [SpanAttributes.COST_NUM_TOKENS]: 150,
    };
    const results: Record<string, ReactNode> = {};

    processTokenAttributes(attributes, results);

    const resultKey = Object.keys(results)[0];
    expect(Object.keys(results)).toHaveLength(1);
    expect(resultKey).toBe('Token count (Usage)');

    render(results[resultKey]);

    expect(screen.getByText('Prompt')).toBeInTheDocument();
    expect(screen.getByText('- 100')).toBeInTheDocument();

    expect(screen.getByText('Completion')).toBeInTheDocument();
    expect(screen.getByText('- 50')).toBeInTheDocument();

    expect(screen.getByText('Total')).toBeInTheDocument();
    expect(screen.getByText('- 150')).toBeInTheDocument();
  });

  it('should remove processed token attributes from original attributes object', () => {
    const attributes = {
      [SpanAttributes.COST_NUM_PROMPT_TOKENS]: 100,
      [SpanAttributes.COST_NUM_COMPLETION_TOKENS]: 50,
      'unrelated.attribute': 'value',
    };
    const results: Record<string, ReactNode> = {};

    processTokenAttributes(attributes, results);

    expect(attributes).toEqual({
      'unrelated.attribute': 'value',
    });
    const resultKey = Object.keys(results)[0];
    render(results[resultKey]);

    expect(screen.getByText('Prompt')).toBeInTheDocument();
    expect(screen.getByText('- 100')).toBeInTheDocument();

    expect(screen.getByText('Completion')).toBeInTheDocument();
    expect(screen.getByText('- 50')).toBeInTheDocument();

    expect(screen.queryByText('attribute')).not.toBeInTheDocument();
  });

  it('should skip null token attributes', () => {
    const attributes = {
      [SpanAttributes.COST_NUM_PROMPT_TOKENS]: null,
      [SpanAttributes.COST_NUM_COMPLETION_TOKENS]: 50,
    };
    const results: Record<string, ReactNode> = {};

    processTokenAttributes(attributes, results);

    expect(Object.keys(results)).toHaveLength(1);
    const resultKey = Object.keys(results)[0];
    render(results[resultKey]);

    expect(screen.queryByText('Prompt')).not.toBeInTheDocument();

    expect(screen.getByText('Completion')).toBeInTheDocument();
    expect(screen.getByText('- 50')).toBeInTheDocument();
  });
});
