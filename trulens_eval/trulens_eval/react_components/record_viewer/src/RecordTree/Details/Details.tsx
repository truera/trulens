import NodeDetails from '@/RecordTree/Details/NodeDetails';
import RootDetails from '@/RecordTree/Details/RootDetails';
import { StackTreeNode } from '@/utils/StackTreeNode';
import { RecordJSONRaw } from '@/utils/types';

type DetailsProps = {
  selectedNode: StackTreeNode;
  recordJSON: RecordJSONRaw;
};

export default function Details({ selectedNode, recordJSON }: DetailsProps) {
  if (!selectedNode) return <>Node not found.</>;

  if (selectedNode.isRoot) return <RootDetails root={selectedNode} recordJSON={recordJSON} />;

  return <NodeDetails selectedNode={selectedNode} recordJSON={recordJSON} />;
}
