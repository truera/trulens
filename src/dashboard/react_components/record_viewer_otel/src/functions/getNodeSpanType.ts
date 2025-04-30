import { StackTreeNode } from '@/types/StackTreeNode';
import { getSpanTypeTitle } from './getSpanTypeTitle';
import { SpanAttributes } from '@/constants/span';

export const getNodeSpanType = (node: StackTreeNode) => {
  return getSpanTypeTitle(node.attributes[SpanAttributes.SPAN_TYPE]);
};
