import { StackTreeNode } from '@/types/StackTreeNode';

export const getDefaultExpandedItems = (node: StackTreeNode, maxDepth: number): string[] => {
  const expanded: string[] = [];

  const traverse = (current: StackTreeNode, depth: number) => {
    if (depth < maxDepth) {
      expanded.push(current.id);
      current.children.forEach((child) => traverse(child, depth + 1));
    }
  };

  traverse(node, 0);
  return expanded;
};


