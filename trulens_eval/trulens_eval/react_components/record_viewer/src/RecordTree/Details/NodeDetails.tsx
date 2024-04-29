import LLMDetails from '@/RecordTree/Details/NodeSpecificDetails/LLMDetails';
import NodeDetailsContainer from '@/RecordTree/Details/NodeSpecificDetails/NodeDetailsContainer';
import { CommonDetailsProps } from '@/RecordTree/Details/NodeSpecificDetails/types';
import { SpanLLM, SpanRetriever } from '@/utils/Span';
import { StackTreeNode } from '@/utils/StackTreeNode';

import RetrieverDetails from './NodeSpecificDetails/RetrieverDetails';

export default function NodeDetails({ selectedNode, recordJSON }: CommonDetailsProps) {
  const { span } = selectedNode;

  if (!span) return <NodeDetailsContainer selectedNode={selectedNode} recordJSON={recordJSON} />;
  if (span instanceof SpanLLM)
    return <LLMDetails selectedNode={selectedNode as StackTreeNode<SpanLLM>} recordJSON={recordJSON} />;

  if (span instanceof SpanRetriever)
    return <RetrieverDetails selectedNode={selectedNode as StackTreeNode<SpanRetriever>} recordJSON={recordJSON} />;

  return <NodeDetailsContainer selectedNode={selectedNode} recordJSON={recordJSON} />;
}
