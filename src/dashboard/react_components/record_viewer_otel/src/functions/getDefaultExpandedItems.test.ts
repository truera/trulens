import { describe, expect, it } from 'vitest';
import { getDefaultExpandedItems } from './getDefaultExpandedItems';
import {
  mockNestedNode,
  mockDeepNode,
  mockMultipleChildrenNode,
  mockNodeWithOrphanedChildren,
} from '@/__testing__/nodes';

describe('getDefaultExpandedItems (otel)', () => {
  const expectNoDuplicates = (arr: string[]) => {
    expect(new Set(arr).size).toBe(arr.length);
  };

  it('returns empty for maxDepth=0', () => {
    expect(getDefaultExpandedItems(mockNestedNode, 0)).toEqual([]);
  });

  it('returns only root for maxDepth=1', () => {
    expect(getDefaultExpandedItems(mockNestedNode, 1)).toEqual([mockNestedNode.id]);
  });

  it('returns preorder: root then its first-level children for maxDepth=2', () => {
    const expanded = getDefaultExpandedItems(mockNestedNode, 2);
    const childIds = mockNestedNode.children.map((c) => c.id);
    expect(expanded).toEqual([mockNestedNode.id, ...childIds]);
    expectNoDuplicates(expanded);
  });

  it('deep chain includes exactly first D nodes', () => {
    const expanded = getDefaultExpandedItems(mockDeepNode, 2);
    expect(expanded).toEqual([mockDeepNode.id, mockDeepNode.children[0].id]);
  });

  it('wide tree includes all first-level children when maxDepth=2', () => {
    const expanded = getDefaultExpandedItems(mockMultipleChildrenNode, 2);
    const expected = [
      mockMultipleChildrenNode.id,
      ...mockMultipleChildrenNode.children.map((c) => c.id),
    ];
    expect(new Set(expanded)).toEqual(new Set(expected));
  });

  it('orphan container and its child expanded when depth allows', () => {
    const expanded = getDefaultExpandedItems(mockNodeWithOrphanedChildren, 3);
    const orphanContainer = mockNodeWithOrphanedChildren.children.find((c) => c.children.length > 0)!;
    expect(expanded).toContain(mockNodeWithOrphanedChildren.id);
    expect(expanded).toContain(orphanContainer.id);
    expect(expanded).toContain(orphanContainer.children[0].id);
  });

  it('does not mutate input tree', () => {
    const before = JSON.stringify(mockNestedNode);
    getDefaultExpandedItems(mockNestedNode, 2);
    const after = JSON.stringify(mockNestedNode);
    expect(after).toBe(before);
  });
});
