import { Grid, Stack, Typography } from '@mui/material';

import JSONViewer from '@/JSONViewer';
import LabelAndValue from '@/LabelAndValue';
import Panel from '@/Panel';
import NodeDetailsContainer from '@/RecordTree/Details/NodeSpecificDetails/NodeDetailsContainer';
import { CommonDetailsProps } from '@/RecordTree/Details/NodeSpecificDetails/types';
import Section from '@/RecordTree/Details/Section';
import { SpanRetriever } from '@/utils/Span';

type RetrieverDetailsProps = CommonDetailsProps<SpanRetriever>;

export default function RetrieverDetails({ selectedNode, recordJSON }: RetrieverDetailsProps) {
  const { span } = selectedNode;

  if (!span) return <NodeDetailsContainer selectedNode={selectedNode} recordJSON={recordJSON} />;

  const {
    inputText,
    inputEmbedding,
    distanceType,
    numContexts,
    retrievedContexts,
    retrievedScores,
    retrievedEmbeddings,
  } = span;

  return (
    <NodeDetailsContainer
      selectedNode={selectedNode}
      recordJSON={recordJSON}
      labels={
        <>
          <LabelAndValue label="Distance type" value={<Typography>{distanceType ?? 'N/A'}</Typography>} />
          <LabelAndValue label="Number of contexts" value={<Typography>{numContexts ?? 'N/A'}</Typography>} />
        </>
      }
    >
      <Grid item xs={12}>
        <Panel header="Retriever I/O">
          <Stack gap={2}>
            <Section title="Input text">
              <Typography>{inputText ?? 'N/A'}</Typography>
            </Section>

            <Section title="Input embedding">
              {inputEmbedding ? <JSONViewer src={inputEmbedding} /> : <Typography>N/A</Typography>}
            </Section>

            <Section title="Retrieved contexts">
              {retrievedContexts?.length ? <JSONViewer src={retrievedContexts} /> : <Typography>N/A</Typography>}
            </Section>

            <Section title="Retrieved scores">
              {retrievedScores?.length ? <JSONViewer src={retrievedScores} /> : <Typography>N/A</Typography>}
            </Section>

            <Section title="Retrieved embeddings">
              {retrievedEmbeddings?.length ? <JSONViewer src={retrievedEmbeddings} /> : <Typography>N/A</Typography>}
            </Section>
          </Stack>
        </Panel>
      </Grid>
    </NodeDetailsContainer>
  );
}
