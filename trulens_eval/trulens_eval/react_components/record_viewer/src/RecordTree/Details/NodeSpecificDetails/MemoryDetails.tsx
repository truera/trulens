import { Grid, Stack, Typography } from '@mui/material';

import LabelAndValue from '@/LabelAndValue';
import Panel from '@/Panel';
import NodeDetailsContainer from '@/RecordTree/Details/NodeSpecificDetails/NodeDetailsContainer';
import { CommonDetailsProps } from '@/RecordTree/Details/NodeSpecificDetails/types';
import Section from '@/RecordTree/Details/Section';
import { SpanMemory } from '@/utils/Span';

type MemoryDetailsProps = CommonDetailsProps<SpanMemory>;

export default function MemoryDetails({ selectedNode, recordJSON }: MemoryDetailsProps) {
  const { span } = selectedNode;

  if (!span) return <NodeDetailsContainer selectedNode={selectedNode} recordJSON={recordJSON} />;

  const { memoryType, remembered } = span;

  return (
    <NodeDetailsContainer
      selectedNode={selectedNode}
      recordJSON={recordJSON}
      labels={<LabelAndValue label="Memory type" value={<Typography>{memoryType ?? 'N/A'}</Typography>} />}
    >
      <Grid item xs={12}>
        <Panel header="Memory details">
          <Stack gap={2}>
            <Section title="Remembered text">
              <Typography>{remembered ?? 'N/A'}</Typography>
            </Section>
          </Stack>
        </Panel>
      </Grid>
    </NodeDetailsContainer>
  );
}
