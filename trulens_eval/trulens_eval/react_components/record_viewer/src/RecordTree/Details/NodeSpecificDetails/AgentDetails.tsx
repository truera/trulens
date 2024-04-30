import { Typography } from '@mui/material';

import LabelAndValue from '@/LabelAndValue';
import NodeDetailsContainer from '@/RecordTree/Details/NodeSpecificDetails/NodeDetailsContainer';
import { CommonDetailsProps } from '@/RecordTree/Details/NodeSpecificDetails/types';
import { SpanAgent } from '@/utils/Span';

type AgentDetailsProps = CommonDetailsProps<SpanAgent>;

export default function AgentDetails({ selectedNode, recordJSON }: AgentDetailsProps) {
  const { span } = selectedNode;

  if (!span) return <NodeDetailsContainer selectedNode={selectedNode} recordJSON={recordJSON} />;

  const { description } = span;

  return (
    <NodeDetailsContainer
      selectedNode={selectedNode}
      recordJSON={recordJSON}
      labels={<LabelAndValue label="Description" value={<Typography>{description ?? 'N/A'}</Typography>} />}
    />
  );
}
