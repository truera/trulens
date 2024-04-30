import { Grid, Stack, Typography } from '@mui/material';
import { PropsWithChildren, ReactElement } from 'react';

import JSONViewer from '@/JSONViewer';
import LabelAndValue from '@/LabelAndValue';
import Panel from '@/Panel';
import { CommonDetailsProps } from '@/RecordTree/Details/NodeSpecificDetails/types';
import Section from '@/RecordTree/Details/Section';
import { summarySx } from '@/RecordTree/Details/styles';
import TracePanel from '@/RecordTree/Details/TracePanel';
import { toHumanSpanType } from '@/utils/Span';

type DetailsProps = PropsWithChildren<
  CommonDetailsProps & {
    labels?: ReactElement;
  }
>;

export default function NodeDetailsContainer({ selectedNode, recordJSON, children, labels }: DetailsProps) {
  const { timeTaken: nodeTime, raw, selector, span } = selectedNode;

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
        <LabelAndValue label="Span type" value={<Typography>{toHumanSpanType(span?.type)}</Typography>} />
        <LabelAndValue label="Time taken" value={<Typography>{nodeTime} ms</Typography>} />
        {labels}
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

        {children}

        <Grid item xs={12}>
          <TracePanel recordJSON={recordJSON} />
        </Grid>
      </Grid>
    </>
  );
}
