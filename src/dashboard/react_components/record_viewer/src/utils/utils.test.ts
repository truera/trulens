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

// Append tests for getDefaultExpandedItems
import { getDefaultExpandedItems } from '@/utils/utils';
import { StackTreeNode } from '@/utils/StackTreeNode';

const makeNode = (name: string, children: StackTreeNode[] = []): StackTreeNode =>
  new StackTreeNode({ name, children, parentNodes: [] });

const linkParents = (node: StackTreeNode, parents: StackTreeNode[] = []) => {
  node.parentNodes = parents;
  node.children.forEach((child) => linkParents(child, [...parents, node]));
};

const expectNoDuplicates = (arr: string[]) => {
  expect(new Set(arr).size).toBe(arr.length);
};

describe('getDefaultExpandedItems (legacy)', () => {
  test('returns empty for maxDepth=0', () => {
    const root = makeNode('root');
    expect(getDefaultExpandedItems(root, 0)).toEqual([]);
  });

  test('returns only root when maxDepth=1', () => {
    const root = makeNode('root');
    linkParents(root);
    const expanded = getDefaultExpandedItems(root, 1);
    expect(expanded).toEqual([root.nodeId]);
  });

  test('returns root and first level when maxDepth=2 (preorder)', () => {
    const childA = makeNode('A');
    const childB = makeNode('B');
    const root = makeNode('root', [childA, childB]);
    linkParents(root);
    const expanded = getDefaultExpandedItems(root, 2);
    expect(new Set(expanded)).toEqual(new Set([root.nodeId, childA.nodeId, childB.nodeId]));
    expectNoDuplicates(expanded);
  });

  test('deep chain includes exactly first D nodes', () => {
    const n3 = makeNode('n3');
    const n2 = makeNode('n2', [n3]);
    const n1 = makeNode('n1', [n2]);
    linkParents(n1);
    const expanded = getDefaultExpandedItems(n1, 2);
    expect(expanded).toEqual([n1.nodeId, n2.nodeId]);
  });

  test('does not mutate input tree', () => {
    const child = makeNode('child');
    const root = makeNode('root', [child]);
    linkParents(root);
    const safeStringify = (n: StackTreeNode): string =>
      JSON.stringify(n, (key, value) => (key === 'parentNodes' ? undefined : value));
    const before = safeStringify(root);
    getDefaultExpandedItems(root, 2);
    const after = safeStringify(root);
    expect(after).toBe(before);
  });
});
