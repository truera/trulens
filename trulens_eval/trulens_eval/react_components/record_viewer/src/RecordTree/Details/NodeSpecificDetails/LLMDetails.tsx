import { Typography } from '@mui/material';

import LabelAndValue from '@/LabelAndValue';
import NodeDetailsContainer from '@/RecordTree/Details/NodeSpecificDetails/NodeDetailsContainer';
import { CommonDetailsProps } from '@/RecordTree/Details/NodeSpecificDetails/types';
import { SpanLLM } from '@/utils/Span';

type LLMDetailsProps = CommonDetailsProps<SpanLLM>;

export default function LLMDetails({ selectedNode, recordJSON }: LLMDetailsProps) {
  return (
    <NodeDetailsContainer
      selectedNode={selectedNode}
      recordJSON={recordJSON}
      labels={
        <LabelAndValue label="Model name" value={<Typography>{selectedNode.span?.modelName ?? 'N/A'}</Typography>} />
      }
    />
  );
}
