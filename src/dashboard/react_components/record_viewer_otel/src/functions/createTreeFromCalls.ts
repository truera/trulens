import { SpanAttributes, SpanType } from '@/constants/span';
import { StackTreeNode } from '@/types/StackTreeNode';
import { Span } from '@/types/Span';

export const createTreeFromCalls = (spans: Span[]) => {
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

  const nodeParentMap = new Map<string, StackTreeNode[]>();
  nodes.forEach((node) => {
    if (!nodeParentMap.has(node.parentId)) {
      nodeParentMap.set(node.parentId, []);
    }
    nodeParentMap.get(node.parentId)?.push(node);
  });

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
