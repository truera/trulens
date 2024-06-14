import { Span } from '@/utils/Span';
import { StackTreeNode } from '@/utils/StackTreeNode';
import { RecordJSONRaw } from '@/utils/types';

export type CommonDetailsProps<SpanType extends Span = Span> = {
  selectedNode: StackTreeNode<SpanType>;
  recordJSON: RecordJSONRaw;
};
