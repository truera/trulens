import { Box, Grid, Stack, Typography } from '@mui/material';
import { RecordJSONRaw, StackTreeNode } from '../../utils/types';
import { getStartAndEndTimesForNode } from '../../utils/treeUtils';
import Panel from '../../Panel/Panel';
import LabelAndValue from '../../LabelAndValue/LabelAndValue';

import Section from './Section';
import { summarySx } from './styles';
import ReactJson from '@microlink/react-json-view';

type DetailsProps = {
  selectedNode: StackTreeNode;
  recordJSON: RecordJSONRaw;
};

export default function NodeDetails({ selectedNode, recordJSON }: DetailsProps) {
  const { timeTaken: nodeTime } = getStartAndEndTimesForNode(selectedNode);
  const { main_input: traceInput, main_output: traceOutput, main_error: error } = recordJSON;
  const { raw } = selectedNode;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
  const { args, rets } = raw ?? {};

  let returnValueDisplay = <Typography>No return values recorded</Typography>;
  if (rets) {
    if (typeof rets === 'string') returnValueDisplay = <Typography>{rets}</Typography>;
    if (typeof rets === 'object')
      returnValueDisplay = <ReactJson src={rets as object} name={null} style={{ fontSize: '14px' }} />;
  }

  return (
    <>
      <Stack direction="row" sx={summarySx}>
        <LabelAndValue label="Latency" value={<Typography>{nodeTime} ms</Typography>} />
      </Stack>

      <Grid container gap={1}>
        <Grid item xs={12} xl={6}>
          <Panel header="Span I/O">
            <Stack gap={2}>
              <Section title="Arguments">
                {args ? <ReactJson src={args} name={null} style={{ fontSize: '14px' }} /> : 'No arguments recorded.'}
              </Section>

              <Section title="Return values">{returnValueDisplay}</Section>
            </Stack>
          </Panel>
        </Grid>

        <Grid item xs={12} xl={6}>
          <Panel header="Trace I/O">
            <Stack gap={2}>
              <Section title="Input" body={traceInput ?? 'No input found.'} />
              <Section title="Output" body={traceOutput ?? 'No output found.'} />
              {error && <Section title="Error" body={error ?? 'No error found.'} />}
            </Stack>
          </Panel>
        </Grid>
      </Grid>
    </>
  );
}
