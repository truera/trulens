import { StackTreeNode } from './StackTreeNode';

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
  const { endTime: treeEnd } = root;

  const recursiveGetChildrenToRender = (node: StackTreeNode, depth: number) => {
    const { startTime } = node;

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
