import { StackTreeNode } from './types';

/**
 * Gets the start and end times for the tree node provided, defaulting to 0.
 *
 * @param node - the tree node to obtain the start and end times for
 * @returns an object containing the start and end times based on the tree node
 * provided.
 */
export const getStartAndEndTimesForNode = (node: StackTreeNode) => {
  const startTime = node.startTime?.getTime() ?? 0;
  const endTime = node.endTime?.getTime() ?? 0;
  return { startTime, endTime, timeTaken: endTime - startTime };
};

/**
 * Gets the depth of the tree based on the node provided.
 *
 * @param node - the tree node to obtain the depth for
 * @returns depth of tree starting at provided node
 */
export const getTreeDepth = (node: StackTreeNode): number => {
  if (!node.children?.length) return 1;

  return Math.max.apply(null, node.children.map(getTreeDepth)) + 1;
};
