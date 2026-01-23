import { describe, it, expect } from 'vitest';
import { createNodeMap } from './createNodeMap';
import { StackTreeNode } from '@/types/StackTreeNode';
import { createStackTreeNode } from '@/__testing__/createStackTreeNode';

describe(createNodeMap.name, () => {
  it('should create a node map from a single node', () => {
    const node: StackTreeNode = createStackTreeNode({
      id: '1',
      children: [],
    });

    const result = createNodeMap(node);

    expect(result).toEqual({ '1': node });
  });

  it('should create a node map from a tree with multiple nodes', () => {
    const child1: StackTreeNode = createStackTreeNode({
      id: '2',
      children: [],
    });

    const child2: StackTreeNode = createStackTreeNode({
      id: '3',
      children: [],
    });

    const root: StackTreeNode = createStackTreeNode({
      id: '1',
      children: [child1, child2],
    });

    const result = createNodeMap(root);

    expect(result).toEqual({
      '1': root,
      '2': child1,
      '3': child2,
    });
  });

  it('should handle nested children correctly', () => {
    const grandchild: StackTreeNode = createStackTreeNode({
      id: '3',
      children: [],
    });

    const child: StackTreeNode = createStackTreeNode({
      id: '2',
      children: [grandchild],
    });

    const root: StackTreeNode = createStackTreeNode({
      id: '1',
      children: [child],
    });

    const result = createNodeMap(root);

    expect(result).toEqual({
      '1': root,
      '2': child,
      '3': grandchild,
    });
  });

  it('should handle an empty tree', () => {
    const node: StackTreeNode = createStackTreeNode({
      id: '',
      children: [],
    });

    const result = createNodeMap(node);

    expect(result).toEqual({ '': node });
  });
});
