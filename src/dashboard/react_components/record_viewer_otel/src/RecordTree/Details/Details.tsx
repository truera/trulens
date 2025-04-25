import NodeDetails from '@/RecordTree/Details/NodeDetails';
import { StackTreeNode } from '@/utils/StackTreeNode';

type DetailsProps = {
  selectedNode: StackTreeNode;
};

export default function Details({ selectedNode }: DetailsProps) {
  if (!selectedNode) return <>Node not found.</>;

  return <NodeDetails selectedNode={selectedNode} />;
}
