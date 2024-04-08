import { useEffect, useState } from 'react';
import ReactJson from '@microlink/react-json-view';
import KeyboardArrowDownRounded from '@mui/icons-material/KeyboardArrowDownRounded';
import KeyboardArrowUpRounded from '@mui/icons-material/KeyboardArrowUpRounded';
import { Box, Divider, Stack, Typography } from '@mui/material';
import { SimpleTreeView } from '@mui/x-tree-view';
import { Streamlit } from 'streamlit-component-lib';
import { RecordJSONRaw, StackTreeNode } from '../../utils/types';
import { getStartAndEndTimesForNode } from '../../utils/treeUtils';
import RecordTreeCellRecursive from './RecordTreeCellRecursive';
import Panel from '../../Panel/Panel';
import LabelAndValue from '../../LabelAndValue/LabelAndValue';
import { Tabs, Tab } from '../../Tabs';
import { ROOT_NODE_ID } from '../../utils/utils';

type RootDetailsProps = {
  root: StackTreeNode;
  recordJSON: RecordJSONRaw;
};

export default function RootDetails({ root, recordJSON }: RootDetailsProps) {
  const { timeTaken: nodeTime } = getStartAndEndTimesForNode(root);
  const { main_input: traceInput, main_output: traceOutput, main_error: error } = recordJSON;

  return (
    <>
      <Stack
        direction="row"
        sx={{
          border: ({ palette }) => `1px solid ${palette.grey[300]}`,
          pl: 2,
          py: 1,
          borderRadius: 0.5,
          width: 'fit-content',
        }}
      >
        <LabelAndValue label="Latency" value={<Typography>{nodeTime} ms</Typography>} />
      </Stack>

      <Box>
        <Panel header="Trace I/O">
          <Stack gap={2}>
            <Stack gap={1}>
              <Typography variant="body2" fontWeight="bold">
                Input
              </Typography>

              <Typography>{traceInput ?? 'No input found.'}</Typography>
            </Stack>

            <Stack gap={1}>
              <Typography variant="body2" fontWeight="bold">
                Output
              </Typography>

              <Typography>{traceOutput ?? 'No output found'} </Typography>
            </Stack>

            {error && (
              <Stack gap={1}>
                <Typography variant="body2" fontWeight="bold">
                  Error
                </Typography>

                <Typography>{error ?? 'No error found'} </Typography>
              </Stack>
            )}
          </Stack>
        </Panel>
      </Box>
    </>
  );
}
