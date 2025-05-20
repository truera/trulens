import { Stack, Typography } from '@mui/material';

import LabelAndValue from '@/LabelAndValue';
import { summarySx } from '@/Details/styles';
import { StackTreeNode } from '@/types/StackTreeNode';
import { formatDuration } from '@/functions/formatters';
import { TraceAttributes } from '@/TraceAttributes/TraceAttributes';
import { getSpanTypeTitle } from '@/functions/getSpanTypeTitle';
import { SpanAttributes } from '@/constants/span';

export interface NodeDetailsProps {
  selectedNode?: StackTreeNode;
}

export default function NodeDetails(props: NodeDetailsProps) {
  const { selectedNode } = props;

  if (!selectedNode) return <Typography>Node not found.</Typography>;

  const { timeTaken: nodeTime, attributes } = selectedNode;
  const spanTypeTitle = getSpanTypeTitle(attributes[SpanAttributes.SPAN_TYPE]);
  const inputId = attributes[SpanAttributes.INPUT_ID];

  return (
    <>
      <Stack direction="row" sx={summarySx}>
        <LabelAndValue label="Time taken" value={<Typography>{formatDuration(nodeTime)}</Typography>} />
        {spanTypeTitle !== 'Unknown' && (
          <LabelAndValue label="Span type" value={<Typography>{spanTypeTitle}</Typography>} />
        )}
        {!!inputId && <LabelAndValue label="Input ID" value={<Typography>{inputId}</Typography>} />}
      </Stack>

      <Stack gap={1}>
        <TraceAttributes attributes={selectedNode?.attributes ?? {}} />
      </Stack>
    </>
  );
}
