import { StackTreeNode } from '@/utils/StackTreeNode';
import { Span } from '@/utils/types';

export const createTreeFromCalls = (spans: Span[]) => {
  const nodes = spans
    .filter(
      (span) =>
        span.record_attributes['ai.observability.span_type'] !== 'eval' &&
        span.record_attributes['ai.observability.span_type'] !== 'eval_root'
    )
    .map((span) => {
      return new StackTreeNode({
        name: span.record.name,
        startTime: span.start_timestamp,
        endTime: span.timestamp,
        raw: span.record_attributes,
        id: span.trace.span_id,
        parentId: span.trace.parent_id,
      });
    });

  const nodeParentMap = new Map<string, StackTreeNode[]>();
  nodes.forEach((node) => {
    if (!nodeParentMap.has(node.parentId)) {
      nodeParentMap.set(node.parentId, []);
    }
    nodeParentMap.get(node.parentId)?.push(node);
  });

  console.log({ nodeParentMap });

  const roots = nodeParentMap.get('') ?? [];

  if (roots.length === 0) {
    throw new Error('No root node found');
  } else if (roots.length > 1) {
    throw new Error('More than one root node found');
  }

  const tree = roots[0];
  const queue = [tree];

  while (queue.length !== 0) {
    const currNode = queue.pop();
    if (!currNode) continue;

    if (nodeParentMap.has(currNode.id)) {
      currNode.children = nodeParentMap.get(currNode.id) ?? [];
      queue.push(...currNode.children);
    }
  }

  return tree;
};

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

/**
 * Formatting timestamp to display
 *
 * @param timestampInSeconds - timestamp in seconds
 * @returns Human-readable formatted timestamp string
 */
export const formatTime = (timestampInSeconds: number) => {
  if (!timestampInSeconds) return '';

  const jsDate = new Date(timestampInSeconds * 1000);

  return `${jsDate.toLocaleDateString()} ${jsDate.toLocaleTimeString('en-US', {
    hour12: false,
  })}.${timestampInSeconds.toString().padStart(6, '0')}`;
};

/**
 * Formatting duration to display.
 *
 * @param durationInSeconds - duration in seconds
 * @returns Human-readable formatted timestamp duration string
 */
export const formatDuration = (durationInSeconds: number) => {
  if (durationInSeconds === null || durationInSeconds === undefined) return '';

  const { format } = new Intl.NumberFormat(navigator.languages, {
    minimumFractionDigits: 0,
    maximumFractionDigits: 3,
  });

  if (durationInSeconds < 0.001) return `${format(durationInSeconds * 1000_000)} µs`;
  if (durationInSeconds < 1) return `${format(durationInSeconds * 1000)} ms`;

  return `${format(durationInSeconds)} s`;
};
