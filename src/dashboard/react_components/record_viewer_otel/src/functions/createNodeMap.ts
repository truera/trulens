import { StackTreeNode } from '@/types/StackTreeNode';

export const createNodeMap = (node: StackTreeNode) => {
  const result: Record<string, StackTreeNode> = {};

  const queue = [node];

  while (queue.length !== 0) {
    const currNode = queue.pop();

    if (!currNode) continue;

    result[currNode.id] = currNode;

    queue.push(...currNode.children);
  }

  return result;
};
