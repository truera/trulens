import { describe, it, expect } from 'vitest';
import { createTreeFromCalls } from './createTreeFromCalls';
import { Span } from '@/types/Span';
import { SpanAttributes, SpanType } from '@/constants/span';
import { StackTreeNode } from '@/types/StackTreeNode';
import { createSpan } from '@/__testing__/createSpan';
import { ORPHANED_NODES_PARENT_ID } from '@/constants/node';

describe(createTreeFromCalls.name, () => {
  it('should create a tree from spans with a single root node', () => {
    const spans: Span[] = [
      createSpan({
        record_name: 'root',
        start_timestamp: 100,
        timestamp: 200,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '1',
        parent_id: '',
      }),
    ];

    const result = createTreeFromCalls(spans);

    expect(result).toBeInstanceOf(StackTreeNode);
    expect(result.id).toBe('1');
    expect(result.name).toBe('root');
    expect(result.children).toEqual([]);
  });

  it('should create a tree with multiple children', () => {
    const spans: Span[] = [
      createSpan({
        record_name: 'root',
        start_timestamp: 100,
        timestamp: 300,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '1',
        parent_id: '',
      }),
      createSpan({
        record_name: 'child1',
        start_timestamp: 120,
        timestamp: 200,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '2',
        parent_id: '1',
      }),
      createSpan({
        record_name: 'child2',
        start_timestamp: 220,
        timestamp: 280,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '3',
        parent_id: '1',
      }),
    ];

    const result = createTreeFromCalls(spans);

    expect(result.id).toBe('1');
    expect(result.children.length).toBe(2);
    expect(result.children[0].id).toBe('2');
    expect(result.children[0].name).toBe('child1');
    expect(result.children[1].id).toBe('3');
    expect(result.children[1].name).toBe('child2');
  });

  it('should create a tree with nested children', () => {
    const spans: Span[] = [
      createSpan({
        record_name: 'root',
        start_timestamp: 100,
        timestamp: 400,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '1',
        parent_id: '',
      }),
      createSpan({
        record_name: 'child',
        start_timestamp: 150,
        timestamp: 350,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '2',
        parent_id: '1',
      }),
      createSpan({
        record_name: 'grandchild',
        start_timestamp: 200,
        timestamp: 300,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '3',
        parent_id: '2',
      }),
    ];

    const result = createTreeFromCalls(spans);

    expect(result.id).toBe('1');
    expect(result.children.length).toBe(1);
    expect(result.children[0].id).toBe('2');
    expect(result.children[0].children.length).toBe(1);
    expect(result.children[0].children[0].id).toBe('3');
  });

  it('should filter out EVAL and EVAL_ROOT span types', () => {
    const spans: Span[] = [
      createSpan({
        record_name: 'root',
        start_timestamp: 100,
        timestamp: 300,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '1',
        parent_id: '',
      }),
      createSpan({
        record_name: 'eval',
        start_timestamp: 150,
        timestamp: 250,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: SpanType.EVAL },
        span_id: '2',
        parent_id: '1',
      }),
      createSpan({
        record_name: 'eval_root',
        start_timestamp: 150,
        timestamp: 250,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: SpanType.EVAL_ROOT },
        span_id: '3',
        parent_id: '1',
      }),
    ];

    const result = createTreeFromCalls(spans);

    expect(result.id).toBe('1');
    expect(result.children.length).toBe(0);
  });

  it('should throw error when no root node is found', () => {
    const spans: Span[] = [
      createSpan({
        record_name: 'not_root',
        start_timestamp: 100,
        timestamp: 200,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '1',
        parent_id: 'non_existent',
      }),
    ];

    expect(() => createTreeFromCalls(spans)).toThrow('No root node found');
  });

  it('should throw error when more than one root node is found', () => {
    const spans: Span[] = [
      createSpan({
        record_name: 'root1',
        start_timestamp: 100,
        timestamp: 200,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '1',
        parent_id: '',
      }),
      createSpan({
        record_name: 'root2',
        start_timestamp: 300,
        timestamp: 400,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '2',
        parent_id: '',
      }),
    ];

    expect(() => createTreeFromCalls(spans)).toThrow('Multiple root nodes found');
  });

  it('should handle spans that are not in chronological order', () => {
    const spans: Span[] = [
      createSpan({
        record_name: 'child',
        start_timestamp: 150,
        timestamp: 250,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '2',
        parent_id: '1',
      }),
      createSpan({
        record_name: 'root',
        start_timestamp: 100,
        timestamp: 300,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '1',
        parent_id: '',
      }),
    ];

    const result = createTreeFromCalls(spans);

    expect(result.id).toBe('1');
    expect(result.children.length).toBe(1);
    expect(result.children[0].id).toBe('2');
  });

  it('should only show the root node when spans form disconnected trees', () => {
    const spans: Span[] = [
      createSpan({
        record_name: 'root',
        start_timestamp: 100,
        timestamp: 200,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '1',
        parent_id: '',
      }),
      createSpan({
        record_name: 'disconnected',
        start_timestamp: 300,
        timestamp: 400,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '2',
        parent_id: 'non_existent',
      }),

      createSpan({
        record_name: 'disconnected',
        start_timestamp: 300,
        timestamp: 400,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '3',
        parent_id: '2',
      }),

      createSpan({
        record_name: 'disconnected',
        start_timestamp: 300,
        timestamp: 400,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '4',
        parent_id: 'non_existent',
      }),
    ];

    const result = createTreeFromCalls(spans);
    expect(result.id).toBe('1');
    expect(result.children.length).toBe(1);
    expect(result.children[0].id).toBe(ORPHANED_NODES_PARENT_ID);
    expect(result.children[0].children.length).toBe(2);
    expect(result.children[0].children[0].id).toBe('2');
    expect(result.children[0].children[0].children.length).toBe(1);
    expect(result.children[0].children[0].children[0].id).toBe('3');
    expect(result.children[0].children[1].id).toBe('4');
  });

  it('should correctly process spans with multiple levels of nesting', () => {
    const spans: Span[] = [
      createSpan({
        record_name: 'root',
        start_timestamp: 100,
        timestamp: 500,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '1',
        parent_id: '',
      }),
      createSpan({
        record_name: 'child1',
        start_timestamp: 150,
        timestamp: 450,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '2',
        parent_id: '1',
      }),
      createSpan({
        record_name: 'grandchild1',
        start_timestamp: 200,
        timestamp: 300,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '3',
        parent_id: '2',
      }),
      createSpan({
        record_name: 'grandchild2',
        start_timestamp: 350,
        timestamp: 400,
        record_attributes: { [SpanAttributes.SPAN_TYPE]: 'OTHER' },
        span_id: '4',
        parent_id: '2',
      }),
    ];

    const result = createTreeFromCalls(spans);

    expect(result.id).toBe('1');
    expect(result.children.length).toBe(1);
    expect(result.children[0].id).toBe('2');
    expect(result.children[0].children.length).toBe(2);
    expect(result.children[0].children[0].id).toBe('3');
    expect(result.children[0].children[1].id).toBe('4');
  });
});
