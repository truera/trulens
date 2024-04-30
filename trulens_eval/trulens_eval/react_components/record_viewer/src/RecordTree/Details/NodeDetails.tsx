import AgentDetails from '@/RecordTree/Details/NodeSpecificDetails/AgentDetails';
import EmbeddingDetails from '@/RecordTree/Details/NodeSpecificDetails/EmbeddingDetails';
import LLMDetails from '@/RecordTree/Details/NodeSpecificDetails/LLMDetails';
import MemoryDetails from '@/RecordTree/Details/NodeSpecificDetails/MemoryDetails';
import NodeDetailsContainer from '@/RecordTree/Details/NodeSpecificDetails/NodeDetailsContainer';
import RerankerDetails from '@/RecordTree/Details/NodeSpecificDetails/RerankerDetails';
import RetrieverDetails from '@/RecordTree/Details/NodeSpecificDetails/RetrieverDetails';
import ToolDetails from '@/RecordTree/Details/NodeSpecificDetails/ToolDetails';
import { CommonDetailsProps } from '@/RecordTree/Details/NodeSpecificDetails/types';
import { SpanAgent, SpanEmbedding, SpanLLM, SpanMemory, SpanReranker, SpanRetriever, SpanTool } from '@/utils/Span';
import { StackTreeNode } from '@/utils/StackTreeNode';

export default function NodeDetails({ selectedNode, recordJSON }: CommonDetailsProps) {
  const { span } = selectedNode;

  if (span instanceof SpanAgent)
    return <AgentDetails selectedNode={selectedNode as StackTreeNode<SpanAgent>} recordJSON={recordJSON} />;

  if (span instanceof SpanEmbedding)
    return <EmbeddingDetails selectedNode={selectedNode as StackTreeNode<SpanEmbedding>} recordJSON={recordJSON} />;

  if (span instanceof SpanLLM)
    return <LLMDetails selectedNode={selectedNode as StackTreeNode<SpanLLM>} recordJSON={recordJSON} />;

  if (span instanceof SpanMemory)
    return <MemoryDetails selectedNode={selectedNode as StackTreeNode<SpanMemory>} recordJSON={recordJSON} />;

  if (span instanceof SpanReranker)
    return <RerankerDetails selectedNode={selectedNode as StackTreeNode<SpanReranker>} recordJSON={recordJSON} />;

  if (span instanceof SpanRetriever)
    return <RetrieverDetails selectedNode={selectedNode as StackTreeNode<SpanRetriever>} recordJSON={recordJSON} />;

  if (span instanceof SpanTool)
    return <ToolDetails selectedNode={selectedNode as StackTreeNode<SpanTool>} recordJSON={recordJSON} />;

  return <NodeDetailsContainer selectedNode={selectedNode} recordJSON={recordJSON} />;
}
