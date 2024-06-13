import { Grid, Stack, Typography } from '@mui/material';

import JSONViewer from '@/JSONViewer';
import LabelAndValue from '@/LabelAndValue';
import Panel from '@/Panel';
import Section from '@/RecordTree/Details/Section';
import { summarySx } from '@/RecordTree/Details/styles';
import TracePanel from '@/RecordTree/Details/TracePanel';
import { StackTreeNode } from '@/utils/StackTreeNode';
import { RecordJSONRaw } from '@/utils/types';
import { formatDuration } from '@/utils/utils';

type DetailsProps = {
  selectedNode: StackTreeNode;
  recordJSON: RecordJSONRaw;
};

export default function NodeDetails({ selectedNode, recordJSON }: DetailsProps) {
  const { timeTaken: nodeTime, raw, selector } = selectedNode;

  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
  const { args, rets } = raw ?? {};

  let returnValueDisplay = <Typography>No return values recorded</Typography>;
  if (rets) {
    if (typeof rets === 'string') returnValueDisplay = <Typography>{rets}</Typography>;
    if (typeof rets === 'object') returnValueDisplay = <JSONViewer src={rets as object} />;
  }

  return (
    <>
      <Stack direction="row" sx={summarySx}>
        <LabelAndValue label="Time taken" value={<Typography>{formatDuration(nodeTime)}</Typography>} />
      </Stack>

      <Grid container gap={1}>
        <Grid item xs={12}>
          <Panel header="Span I/O">
            <Stack gap={2}>
              <Section title="Arguments" subtitle={selector ? `${selector}.args` : undefined}>
                {args ? <JSONViewer src={args} /> : 'No arguments recorded.'}
              </Section>

              <Section title="Return values" subtitle={selector ? `${selector}.rets` : undefined}>
                {returnValueDisplay}
              </Section>
            </Stack>
          </Panel>
        </Grid>

        <Grid item xs={12}>
          <TracePanel recordJSON={recordJSON} />
        </Grid>
      </Grid>
    </>
  );
}
