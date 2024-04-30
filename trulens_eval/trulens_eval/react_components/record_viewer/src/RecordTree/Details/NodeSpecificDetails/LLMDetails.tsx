import { Grid, Stack, Typography } from '@mui/material';

import JSONViewer from '@/JSONViewer';
import LabelAndValue from '@/LabelAndValue';
import Panel from '@/Panel';
import NodeDetailsContainer from '@/RecordTree/Details/NodeSpecificDetails/NodeDetailsContainer';
import { CommonDetailsProps } from '@/RecordTree/Details/NodeSpecificDetails/types';
import Section from '@/RecordTree/Details/Section';
import { SpanLLM } from '@/utils/Span';

type LLMDetailsProps = CommonDetailsProps<SpanLLM>;

export default function LLMDetails({ selectedNode, recordJSON }: LLMDetailsProps) {
  const { span } = selectedNode;

  if (!span) return <NodeDetailsContainer selectedNode={selectedNode} recordJSON={recordJSON} />;

  const { modelName, modelType, temperature, inputMessages, inputTokenCount, outputMessages, outputTokenCount, cost } =
    span;

  return (
    <NodeDetailsContainer
      selectedNode={selectedNode}
      recordJSON={recordJSON}
      labels={
        <>
          <LabelAndValue label="Temperature" value={<Typography>{temperature ?? 'N/A'}</Typography>} />
          <LabelAndValue label="Cost" value={<Typography>{cost ?? 'N/A'}</Typography>} />
          <LabelAndValue label="Model name" value={<Typography>{modelName ?? 'N/A'}</Typography>} />
          <LabelAndValue label="Model type" value={<Typography>{modelType ?? 'N/A'}</Typography>} />
        </>
      }
    >
      <Grid item xs={12}>
        <Panel header="Retriever I/O">
          <Stack gap={2}>
            <Section title="Input messages">
              {inputMessages ? <JSONViewer src={inputMessages} /> : <Typography>N/A</Typography>}
            </Section>

            <Section title="Input token count">
              <Typography>{inputTokenCount ?? 'N/A'}</Typography>
            </Section>

            <Section title="Output messages">
              {outputMessages ? <JSONViewer src={outputMessages} /> : <Typography>N/A</Typography>}
            </Section>

            <Section title="Output token count">
              <Typography>{outputTokenCount ?? 'N/A'}</Typography>
            </Section>
          </Stack>
        </Panel>
      </Grid>
    </NodeDetailsContainer>
  );
}
