import { Stack, Typography } from '@mui/material';

import LabelAndValue from '@/LabelAndValue';
import { summarySx } from '@/RecordTree/Details/styles';
import { StackTreeNode } from '@/utils/StackTreeNode';
import { formatDuration } from '@/utils/utils';
import { TraceAttributes } from '@/TraceAttributes/TraceAttributes';

type DetailsProps = {
  selectedNode: StackTreeNode;
};

export default function NodeDetails(props: DetailsProps) {
  const { selectedNode } = props;
  const { timeTaken: nodeTime } = selectedNode;

  return (
    <>
      <Stack direction="row" sx={summarySx}>
        <LabelAndValue label="Time taken" value={<Typography>{formatDuration(nodeTime)}</Typography>} />
      </Stack>

      <Stack gap={1}>
        <TraceAttributes attributes={selectedNode?.raw ?? {}} />
      </Stack>
    </>
  );
}
