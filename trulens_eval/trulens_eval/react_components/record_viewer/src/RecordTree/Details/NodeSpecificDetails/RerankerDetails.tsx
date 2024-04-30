import { Grid, Stack, Typography } from '@mui/material';

import JSONViewer from '@/JSONViewer';
import LabelAndValue from '@/LabelAndValue';
import Panel from '@/Panel';
import NodeDetailsContainer from '@/RecordTree/Details/NodeSpecificDetails/NodeDetailsContainer';
import { CommonDetailsProps } from '@/RecordTree/Details/NodeSpecificDetails/types';
import Section from '@/RecordTree/Details/Section';
import { SpanReranker } from '@/utils/Span';

type RerankerDetailsProps = CommonDetailsProps<SpanReranker>;

export default function RerankerDetails({ selectedNode, recordJSON }: RerankerDetailsProps) {
  const { span } = selectedNode;

  if (!span) return <NodeDetailsContainer selectedNode={selectedNode} recordJSON={recordJSON} />;

  const { queryText, modelName, topN, inputContextScores, inputContextTexts, outputRanks } = span;

  return (
    <NodeDetailsContainer
      selectedNode={selectedNode}
      recordJSON={recordJSON}
      labels={
        <>
          <LabelAndValue label="Model name" value={<Typography>{modelName ?? 'N/A'}</Typography>} />
          <LabelAndValue label="Top N" value={<Typography>{topN ?? 'N/A'}</Typography>} />
        </>
      }
    >
      <Grid item xs={12}>
        <Panel header="Retriever I/O">
          <Stack gap={2}>
            <Section title="Query text">
              <Typography>{queryText ?? 'N/A'}</Typography>
            </Section>

            <Section title="Input context scores">
              {inputContextScores ? <JSONViewer src={inputContextScores} /> : <Typography>N/A</Typography>}
            </Section>

            <Section title="Input context texts">
              {inputContextTexts ? <JSONViewer src={inputContextTexts} /> : <Typography>N/A</Typography>}
            </Section>

            <Section title="Output ranks">
              {outputRanks ? <JSONViewer src={outputRanks} /> : <Typography>N/A</Typography>}
            </Section>
          </Stack>
        </Panel>
      </Grid>
    </NodeDetailsContainer>
  );
}
