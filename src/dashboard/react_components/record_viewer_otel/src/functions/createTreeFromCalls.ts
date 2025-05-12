import { SpanAttributes, SpanType } from '@/constants/span';
import { StackTreeNode } from '@/types/StackTreeNode';
import { Span } from '@/types/Span';
import { ORPHANED_NODES_PARENT_ID } from '@/constants/node';

export const createTreeFromCalls = (spans: Span[]) => {
  if (spans.length === 0) {
    throw new Error('No spans provided');
  }

  const nodes = spans
    .filter(
      (span) =>
        span.record_attributes[SpanAttributes.SPAN_TYPE] !== SpanType.EVAL &&
        span.record_attributes[SpanAttributes.SPAN_TYPE] !== SpanType.EVAL_ROOT
    )
    .map((span) => {
      return new StackTreeNode({
        name: span.record.name,
        startTime: span.start_timestamp,
        endTime: span.timestamp,
        attributes: span.record_attributes,
        id: span.trace.span_id,
        parentId: span.trace.parent_id,
      });
    });

  return createSpanTreeFromNodes(nodes);
};

/**
 * Creates a tree structure from a list of nodes based on parent-child relationships.
 * Orphaned nodes (those without a valid parent in the tree) are attached to a special
 * orphaned nodes container.
 * @param nodes List of StackTreeNode objects
 * @returns The root node of the created tree
 */
export const createSpanTreeFromNodes = (nodes: StackTreeNode[]) => {
  if (nodes.length === 0) {
    throw new Error('No nodes provided');
  }

  // Map nodes by their IDs for quick lookup
  const nodeMap = new Map<string, StackTreeNode>();
  nodes.forEach((node) => nodeMap.set(node.id, node));

  // Group children by their parentId
  const childrenMap = new Map<string, StackTreeNode[]>();
  nodes.forEach((node) => {
    if (!childrenMap.has(node.parentId)) {
      childrenMap.set(node.parentId, []);
    }
    childrenMap.get(node.parentId)?.push(node);
  });

  // Find the root node
  const rootNodes = nodes.filter((node) => node.attributes[SpanAttributes.SPAN_TYPE] === SpanType.RECORD_ROOT);

  if (rootNodes.length === 0) throw new Error('No root node found');
  if (rootNodes.length > 1) throw new Error('Multiple root nodes found');

  const root = rootNodes[0];

  // Build the tree by assigning children to their parents
  buildTreeRecursive(root, childrenMap);

  // Handle orphaned nodes (nodes whose parent is not in our set)
  const processedIds = new Set<string>();
  collectNodeIds(root, processedIds);

  // Find potential orphaned root nodes (nodes whose parent ID isn't in our node map)
  const orphanedRootCandidates = nodes.filter(
    (node) => !processedIds.has(node.id) && node.parentId && !nodeMap.has(node.parentId)
  );

  // Group orphaned nodes by their parentId to identify subtrees
  const orphanedNodes: StackTreeNode[] = [];

  orphanedRootCandidates.forEach((node) => {
    buildTreeRecursive(node, childrenMap);
    orphanedNodes.push(node);
  });

  // If orphaned nodes exist, attach them to a special container
  if (orphanedNodes.length > 0) {
    const orphanContainer = new StackTreeNode({
      name: 'Orphaned nodes',
      startTime: root.startTime,
      // Have the endTime = startTime to hide the duration in the table view
      endTime: root.startTime,
      attributes: {},
      id: ORPHANED_NODES_PARENT_ID,
      parentId: root.id,
      children: orphanedNodes,
    });

    root.children.push(orphanContainer);
  }

  return root;
};

/**
 * Helper function to recursively build the tree by assigning children to parents
 */
const buildTreeRecursive = (node: StackTreeNode, childrenMap: Map<string, StackTreeNode[]>): void => {
  const children = childrenMap.get(node.id) || [];
  node.children = children;

  children.forEach((child) => {
    buildTreeRecursive(child, childrenMap);
  });
};

/**
 * Helper function to collect all node IDs in the tree
 */
const collectNodeIds = (node: StackTreeNode, idSet: Set<string>): void => {
  idSet.add(node.id);
  node.children.forEach((child) => {
    collectNodeIds(child, idSet);
  });
};
