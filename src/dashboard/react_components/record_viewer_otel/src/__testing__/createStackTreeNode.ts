import { StackTreeNode } from '../types/StackTreeNode';
import { SpanAttributes } from '../types/SpanAttributes';

/**
 * Creates a StackTreeNode with reasonable defaults for missing fields.
 *
 * @param options User-provided fields for the StackTreeNode
 * @returns A new StackTreeNode instance
 */
export function createStackTreeNode(
  options: Partial<{
    children: StackTreeNode[];
    name: string;
    id: string;
    startTime: number;
    endTime: number;
    attributes: SpanAttributes;
    parentId: string;
  }> = {}
): StackTreeNode {
  const now = Date.now();

  return new StackTreeNode({
    children: options.children || [],
    name: options.name ?? 'unnamed-node',
    id: options.id ?? `node-${Math.random().toString(36).substring(2, 11)}`,
    startTime: options.startTime ?? now,
    endTime: options.endTime ?? now + 100, // Default 100ms duration
    attributes: options.attributes || {},
    parentId: options.parentId ?? '',
  });
}
