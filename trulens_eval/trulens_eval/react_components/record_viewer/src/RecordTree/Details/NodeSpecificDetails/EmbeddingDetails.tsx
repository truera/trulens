import { Grid, Stack, Typography } from '@mui/material';

import JSONViewer from '@/JSONViewer';
import LabelAndValue from '@/LabelAndValue';
import Panel from '@/Panel';
import NodeDetailsContainer from '@/RecordTree/Details/NodeSpecificDetails/NodeDetailsContainer';
import { CommonDetailsProps } from '@/RecordTree/Details/NodeSpecificDetails/types';
import Section from '@/RecordTree/Details/Section';
import { SpanEmbedding } from '@/utils/Span';

type EmbeddingDetailsProps = CommonDetailsProps<SpanEmbedding>;

export default function EmbeddingDetails({ selectedNode, recordJSON }: EmbeddingDetailsProps) {
  const { span } = selectedNode;

  if (!span) return <NodeDetailsContainer selectedNode={selectedNode} recordJSON={recordJSON} />;

  const { inputText, modelName, embedding } = span;

  return (
    <NodeDetailsContainer
      selectedNode={selectedNode}
      recordJSON={recordJSON}
      labels={<LabelAndValue label="Model name" value={<Typography>{modelName ?? 'N/A'}</Typography>} />}
    >
      <Grid item xs={12}>
        <Panel header="Embedding details">
          <Stack gap={2}>
            <Section title="Input text">
              <Typography>{inputText ?? 'N/A'}</Typography>
            </Section>

            <Section title="Embedding">
              {embedding ? <JSONViewer src={embedding} /> : <Typography>N/A</Typography>}
            </Section>
          </Stack>
        </Panel>
      </Grid>
    </NodeDetailsContainer>
  );
}
