import { RecordJSONRaw, StackTreeNode } from '../../utils/types';
import { ROOT_NODE_ID } from '../../utils/utils';
import RootDetails from './RootDetails';
import NodeDetails from './NodeDetails';

type DetailsProps = {
  selectedNode: StackTreeNode;
  recordJSON: RecordJSONRaw;
};

export default function Details({ selectedNode, recordJSON }: DetailsProps) {
  if (!selectedNode) return <>Node not found.</>;

  if (selectedNode.nodeId === ROOT_NODE_ID) return <RootDetails root={selectedNode} recordJSON={recordJSON} />;

  return <NodeDetails selectedNode={selectedNode} recordJSON={recordJSON} />;
}
