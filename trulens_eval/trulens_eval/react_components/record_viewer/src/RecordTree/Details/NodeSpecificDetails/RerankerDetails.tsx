import {
  Grid,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';

import LabelAndValue from '@/LabelAndValue';
import Panel from '@/Panel';
import NodeDetailsContainer from '@/RecordTree/Details/NodeSpecificDetails/NodeDetailsContainer';
import { CommonDetailsProps } from '@/RecordTree/Details/NodeSpecificDetails/types';
import Section from '@/RecordTree/Details/Section';
import { SpanReranker } from '@/utils/Span';
import { tableWithBorderSx } from '@/utils/styling';

type RerankerDetailsProps = CommonDetailsProps<SpanReranker>;

export default function RerankerDetails({ selectedNode, recordJSON }: RerankerDetailsProps) {
  const { span } = selectedNode;

  if (!span) return <NodeDetailsContainer selectedNode={selectedNode} recordJSON={recordJSON} />;

  const { queryText, modelName, topN, contexts } = span;

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
        <Panel header="Reranker details">
          <Stack gap={2}>
            <Section title="Query text">
              <Typography>{queryText ?? 'N/A'}</Typography>
            </Section>

            <Section title="Contexts">
              {contexts ? (
                <TableContainer>
                  <Table aria-label="Table of contexts passed to the reranker" size="small" sx={tableWithBorderSx}>
                    <TableHead>
                      <TableRow>
                        <TableCell>Context</TableCell>
                        <TableCell align="right">Input score</TableCell>
                        <TableCell align="right">Output rank</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {contexts.map(({ context, inputScore, outputRank }) => (
                        <TableRow key={context}>
                          <TableCell>{context}</TableCell>
                          <TableCell align="right">{inputScore ?? '-'}</TableCell>
                          <TableCell align="right">{outputRank ?? '-'}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              ) : (
                <Typography>N/A</Typography>
              )}
            </Section>
          </Stack>
        </Panel>
      </Grid>
    </NodeDetailsContainer>
  );
}
