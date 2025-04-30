import NodeDetails from '@/RecordTree/Details/NodeDetails';
import { StackTreeNode } from '@/types/StackTreeNode';
import { Typography } from '@mui/material';

type DetailsProps = {
  selectedNode: StackTreeNode;
};

export default function Details({ selectedNode }: DetailsProps) {
  if (!selectedNode) return <Typography>Node not found.</Typography>;

  return <NodeDetails selectedNode={selectedNode} />;
}
