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

/**
 * Get a list of nodes to be rendered.
 *
 * @param root - the root of the tree to recursively get the nodes for
 * @returns Array of nodes found via DFS
 */
export const getNodesToRender = (root: StackTreeNode) => {
  const children: { node: StackTreeNode; depth: number }[] = [];
  const { endTime: treeEnd } = getStartAndEndTimesForNode(root);

  const recursiveGetChildrenToRender = (node: StackTreeNode, depth: number) => {
    const { startTime } = getStartAndEndTimesForNode(node);

    // Ignore calls that happen after the app time. This is indicative of errors.
    if (startTime >= treeEnd) return;

    children.push({ node, depth });

    if (!node.children) return;

    node.children.forEach((child) => {
      recursiveGetChildrenToRender(child, depth + 1);
    });
  };

  recursiveGetChildrenToRender(root, 0);

  return children;
};
