import { Grid, Stack, Typography } from '@mui/material';
import { RecordJSONRaw, StackTreeNode } from '../../utils/types';
import { getStartAndEndTimesForNode } from '../../utils/treeUtils';
import Panel from '../../Panel/Panel';
import LabelAndValue from '../../LabelAndValue/LabelAndValue';

import Section from './Section';
import { summarySx } from './styles';
import JSONViewer from '../../JSONViewer/JSONViewer';

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
    if (typeof rets === 'object') returnValueDisplay = <JSONViewer src={rets as object} />;
  }

  return (
    <>
      <Stack direction="row" sx={summarySx}>
        <LabelAndValue label="Time taken" value={<Typography>{nodeTime} ms</Typography>} />
      </Stack>

      <Grid container gap={1}>
        <Grid item xs={12} xl={6}>
          <Panel header="Span I/O">
            <Stack gap={2}>
              <Section title="Arguments">{args ? <JSONViewer src={args} /> : 'No arguments recorded.'}</Section>

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
