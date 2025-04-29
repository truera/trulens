import { Stack, Typography } from '@mui/material';

import LabelAndValue from '@/LabelAndValue';
import { summarySx } from '@/RecordTree/Details/styles';
import { StackTreeNode } from '@/types/StackTreeNode';
import { formatDuration } from '@/functions/formatters';
import { TraceAttributes } from '@/TraceAttributes/TraceAttributes';
import { getSpanTypeTitle } from '@/functions/getSpanTypeTitle';
import { SpanAttributes } from '@/constants/span';

type DetailsProps = {
  selectedNode: StackTreeNode;
};

export default function NodeDetails(props: DetailsProps) {
  const { selectedNode } = props;
  const { timeTaken: nodeTime, attributes } = selectedNode;
  const spanTypeTitle = getSpanTypeTitle(attributes[SpanAttributes.SPAN_TYPE]);

  return (
    <>
      <Stack direction="row" sx={summarySx}>
        <LabelAndValue label="Time taken" value={<Typography>{formatDuration(nodeTime)}</Typography>} />
        {spanTypeTitle !== 'Unknown' && (
          <LabelAndValue label="Span type" value={<Typography>{spanTypeTitle} </Typography>} />
        )}
      </Stack>

      <Stack gap={1}>
        <TraceAttributes attributes={selectedNode?.attributes ?? {}} />
      </Stack>
    </>
  );
}
