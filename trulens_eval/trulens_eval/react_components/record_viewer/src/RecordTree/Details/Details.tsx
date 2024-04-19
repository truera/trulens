import { RecordJSONRaw } from '../../utils/types';
import { StackTreeNode } from '../../utils/StackTreeNode';
import RootDetails from './RootDetails';
import NodeDetails from './NodeDetails';

type DetailsProps = {
  selectedNode: StackTreeNode;
  recordJSON: RecordJSONRaw;
};

export default function Details({ selectedNode, recordJSON }: DetailsProps) {
  if (!selectedNode) return <>Node not found.</>;

  if (selectedNode.isRoot) return <RootDetails root={selectedNode} recordJSON={recordJSON} />;

  return <NodeDetails selectedNode={selectedNode} recordJSON={recordJSON} />;
}
