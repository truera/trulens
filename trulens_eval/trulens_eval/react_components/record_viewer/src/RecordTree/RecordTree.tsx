import { useEffect, useState } from 'react';
import ReactJson from '@microlink/react-json-view';
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
import { ROOT_NODE_ID } from '../utils/utils';
import Details from './Details/Details';

type RecordTreeProps = {
  nodeMap: Record<string, StackTreeNode>;
  recordJSON: RecordJSONRaw;
  root: StackTreeNode;
};

export default function RecordTree({ nodeMap, recordJSON, root }: RecordTreeProps) {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedTab, setSelectedTab] = useState<string>('Details');

  const handleItemSelectionToggle = (_event: React.SyntheticEvent, itemId: string, isSelected: boolean) => {
    if (isSelected) {
      setSelectedNodeId(itemId);
    } else {
      setSelectedNodeId(null);
    }
  };
  const selectedNode = selectedNodeId ? nodeMap[selectedNodeId] : root;

  useEffect(() => Streamlit.setComponentValue(selectedNodeId), [selectedNodeId]);

  const { timeTaken: totalTime, startTime: treeStart } = getStartAndEndTimesForNode(root);

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
        defaultSelectedItems={ROOT_NODE_ID}
        defaultExpandedItems={Object.keys(nodeMap) ?? []}
        onItemSelectionToggle={handleItemSelectionToggle}
      >
        <RecordTreeCellRecursive node={root} depth={0} totalTime={totalTime} treeStart={treeStart} />
      </SimpleTreeView>

      <Stack sx={{ flexGrow: 1 }}>
        <Tabs
          value={selectedTab}
          onChange={(_event, value) => setSelectedTab(value)}
          sx={{ borderBottom: ({ palette }) => `1px solid ${palette.grey[300]}` }}
        >
          <Tab label="Details" value="Details" id="Details" />
          <Tab label="Raw JSON" value="Raw JSON" id="Raw JSON" />
        </Tabs>

        <Stack gap={2} sx={{ flexGrow: 1, p: 1, mb: 4 }}>
          {selectedTab === 'Details' ? (
            <Details selectedNode={selectedNode} recordJSON={recordJSON} />
          ) : (
            <ReactJson
              src={selectedNodeId === ROOT_NODE_ID ? recordJSON : selectedNode.raw ?? {}}
              name={null}
              style={{ fontSize: '14px' }}
            />
          )}
        </Stack>
      </Stack>
    </Stack>
  );
}
