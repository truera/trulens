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
import RootDetails from './RootDetails';
import NodeDetails from './NodeDetails';

type DetailsProps = {
  selectedNode: StackTreeNode;
  recordJSON: RecordJSONRaw;
};

export default function Details({ selectedNode, recordJSON }: DetailsProps) {
  if (!selectedNode) return <>Node not found.</>;

  if (selectedNode.nodeId === ROOT_NODE_ID) return <RootDetails root={selectedNode} recordJSON={recordJSON} />;

  return <NodeDetails selectedNode={selectedNode} recordJSON={recordJSON} />;
}
