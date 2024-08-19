import { Stack, SxProps, Theme, Typography } from '@mui/material';

import LabelAndValue from '@/LabelAndValue';
import TracePanel from '@/RecordTree/Details/TracePanel';
import { StackTreeNode } from '@/utils/StackTreeNode';
import { RecordJSONRaw } from '@/utils/types';
import { formatDuration } from '@/utils/utils';

type RootDetailsProps = {
  root: StackTreeNode;
  recordJSON: RecordJSONRaw;
};

export default function RootDetails({ root, recordJSON }: RootDetailsProps) {
  const { timeTaken: nodeTime } = root;

  return (
    <>
      <Stack direction="row" sx={rootDetailsContainerSx}>
        <LabelAndValue label="Latency" value={<Typography>{formatDuration(nodeTime)}</Typography>} />
      </Stack>

      <TracePanel recordJSON={recordJSON} />
    </>
  );
}

const rootDetailsContainerSx: SxProps<Theme> = {
  border: ({ palette }) => `1px solid ${palette.grey[300]}`,
  pl: 2,
  py: 1,
  borderRadius: 0.5,
  width: 'fit-content',
};
