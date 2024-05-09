import { Stack } from '@mui/material';

import Panel from '@/Panel/Panel';
import Section from '@/RecordTree/Details/Section';
import { RecordJSONRaw } from '@/utils/types';

type TracePanelProps = {
  recordJSON: RecordJSONRaw;
};

export default function TracePanel({ recordJSON }: TracePanelProps) {
  const { main_input: traceInput, main_output: traceOutput, main_error: error } = recordJSON;

  return (
    <Panel header="Trace I/O">
      <Stack gap={2}>
        <Section title="Input" subtitle="Select.RecordInput" body={traceInput ?? 'No input found.'} />
        <Section title="Output" subtitle="Select.RecordOutput" body={traceOutput ?? 'No output found.'} />
        {error && <Section title="Error" body={error ?? 'No error found.'} />}
      </Stack>
    </Panel>
  );
}
