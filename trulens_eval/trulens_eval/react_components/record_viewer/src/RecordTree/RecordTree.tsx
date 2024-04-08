import { useEffect, useState } from 'react';
import KeyboardArrowDownRounded from '@mui/icons-material/KeyboardArrowDownRounded';
import KeyboardArrowUpRounded from '@mui/icons-material/KeyboardArrowUpRounded';
import { Box, Divider, Stack, Typography } from '@mui/material';
import { SimpleTreeView } from '@mui/x-tree-view';
import { Streamlit } from 'streamlit-component-lib';
import { RecordJSONRaw, StackTreeNode } from '../utils/types';
import { getStartAndEndTimesForNode } from '../utils/treeUtils';
import RecordTreeCellRecursive from './RecordTreeCellRecursive';
import Panel from '../Panel/Panel';
import LabelAndValue from '../LabelAndValue/LabelAndValue';
import { Tabs, Tab } from '../Tabs';
import ReactJson from '@microlink/react-json-view';

type RecordTreeProps = {
  root: StackTreeNode;
  recordJSON: RecordJSONRaw;
};

export default function RecordTree({ recordJSON, root }: RecordTreeProps) {
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [selectedTab, setSelectedTab] = useState<string>('Details');

  const handleItemSelectionToggle = (_event: React.SyntheticEvent, itemId: string, isSelected: boolean) => {
    if (isSelected) {
      setSelectedNode(itemId);
    } else {
      setSelectedNode(null);
    }
  };

  useEffect(() => Streamlit.setComponentValue(selectedNode), [selectedNode]);

  const { timeTaken: totalTime, startTime: treeStart } = getStartAndEndTimesForNode(root);
  const { main_input: traceInput, main_output: traceOutput, main_error: error } = recordJSON;

  return (
    <Stack
      direction="row"
      divider={<Divider orientation="vertical" flexItem />}
      sx={{ border: ({ palette }) => `1px solid ${palette.grey[300]}`, borderRadius: 0.5 }}
    >
      <SimpleTreeView
        sx={{ p: 1, overflowY: 'auto', minWidth: 400 }}
        slots={{
          collapseIcon: KeyboardArrowUpRounded,
          expandIcon: KeyboardArrowDownRounded,
        }}
        onExpandedItemsChange={() => {
          setTimeout(() => Streamlit.setFrameHeight(), 300);
        }}
        defaultSelectedItems="root-0" // TODO: make constant
        onItemSelectionToggle={handleItemSelectionToggle}
      >
        <RecordTreeCellRecursive node={root} depth={0} totalTime={totalTime} treeStart={treeStart} />
      </SimpleTreeView>

      <Stack>
        <Tabs
          value={selectedTab}
          onChange={(_event, value) => setSelectedTab(value)}
          sx={{ borderBottom: ({ palette }) => `1px solid ${palette.grey[300]}` }}
        >
          <Tab label="Details" value="Details" id="Details" />
          <Tab label="Raw Record JSON" value="Raw JSON" id="Raw JSON" />
        </Tabs>

        <Stack gap={2} sx={{ flexGrow: 1, p: 1, mb: 4 }}>
          {selectedTab === 'Details' ? (
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
                <LabelAndValue label="Latency" value={<Typography>{totalTime} ms</Typography>} />
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
          ) : (
            <ReactJson src={recordJSON} name={null} style={{ fontSize: '14px' }} />
          )}
        </Stack>
      </Stack>
    </Stack>
  );
}
