import { Stack } from '@mui/material';

import Panel from '@/Panel/Panel';
import Section from '@/Details/Section';
import { StackTreeNode } from '@/types/StackTreeNode';
import { SpanAttributes } from '@/constants/span';

type TracePanelProps = {
  root: StackTreeNode;
};

export default function TracePanel({ root }: TracePanelProps) {
  const traceInput = root.attributes[SpanAttributes.RECORD_ROOT_INPUT];
  const traceOutput = root.attributes[SpanAttributes.RECORD_ROOT_OUTPUT];
  const error = root.attributes[SpanAttributes.RECORD_ROOT_ERROR];

  return (
    <Panel header="Record I/O">
      <Stack gap={2}>
        <Section title="Input" body={traceInput ?? 'No input found.'} />
        <Section title="Output" body={traceOutput ?? 'No output found.'} />
        {error && <Section title="Error" body={error ?? 'No error found.'} />}
      </Stack>
    </Panel>
  );
}
