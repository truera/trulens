import { Box, Grid, gridClasses, Stack, Typography } from '@mui/material';
import { useState } from 'react';

import JSONViewer from '@/JSONViewer/JSONViewer';
import RecordTable from '@/RecordTable/RecordTable';
import Details from '@/RecordTree/Details/Details';
import RecordTree from '@/RecordTree/RecordTree';
import { Tab, Tabs } from '@/Tabs';
import { StackTreeNode } from '@/utils/StackTreeNode';
import { AppJSONRaw, RecordJSONRaw } from '@/utils/types';

/**
 * Constants and enums for the view
 */
enum RECORD_CONTENT_TABS {
  DETAILS = 'Details',
  SPAN_JSON = 'Span JSON',
  RECORD_JSON = 'Record JSON',
  APP_JSON = 'App JSON',
  RECORD_METADATA = 'Metadata',
}

const SPAN_TREE_TABS = [RECORD_CONTENT_TABS.DETAILS, RECORD_CONTENT_TABS.SPAN_JSON];

const GENERAL_TABS = [
  RECORD_CONTENT_TABS.RECORD_METADATA,
  RECORD_CONTENT_TABS.RECORD_JSON,
  RECORD_CONTENT_TABS.APP_JSON,
];

enum SPAN_VIEW {
  TREE = 'Tree',
  TIMELINE = 'Timeline',
}

const SPAN_VIEWS = [SPAN_VIEW.TREE, SPAN_VIEW.TIMELINE];

type RecordTreeProps = {
  appJSON: AppJSONRaw;
  nodeMap: Record<string, StackTreeNode>;
  recordJSON: RecordJSONRaw;
  root: StackTreeNode;
};

/**
 * Main entryway into presenting the record info. Holds the user-chosen view for displaying the
 * spans and what information the user wishes to show.
 */
export default function RecordInfo({ appJSON, nodeMap, recordJSON, root }: RecordTreeProps) {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedSpanView, setSelectedSpanView] = useState<SPAN_VIEW>(SPAN_VIEW.TREE);
  const [selectedTab, setSelectedTab] = useState<RECORD_CONTENT_TABS>(RECORD_CONTENT_TABS.DETAILS);

  const selectedNode = selectedNodeId ? nodeMap[selectedNodeId] : root;

  // Changes the right hand panel depending on user selection.
  const getSelectedView = () => {
    if (selectedTab === RECORD_CONTENT_TABS.APP_JSON) {
      return <JSONViewer src={appJSON} />;
    }

    if (selectedTab === RECORD_CONTENT_TABS.SPAN_JSON) {
      return <JSONViewer src={selectedNodeId === root.nodeId ? recordJSON : selectedNode.raw ?? {}} />;
    }

    if (selectedTab === RECORD_CONTENT_TABS.RECORD_JSON) {
      return <JSONViewer src={recordJSON} />;
    }

    if (selectedTab === RECORD_CONTENT_TABS.RECORD_METADATA) {
      const { meta } = recordJSON;
      if (!meta || !Object.keys(meta as object)?.length) return <Typography>No record metadata available.</Typography>;

      if (typeof meta === 'object') {
        return <JSONViewer src={meta as object} />;
      }

      return (
        <Typography>Invalid metadata type. Expected a dictionary but got {String(meta) ?? 'unknown object'}</Typography>
      );
    }

    return <Details selectedNode={selectedNode} recordJSON={recordJSON} />;
  };

  const isTimeline = selectedSpanView === SPAN_VIEW.TIMELINE;

  return (
    <Grid
      container
      sx={{
        border: ({ palette }) => `1px solid ${palette.grey[300]}`,
        borderRadius: ({ spacing }) => spacing(0.5),
        [`& > .${gridClasses.item}`]: {
          border: ({ palette }) => `1px solid ${palette.grey[300]}`,
        },
      }}
    >
      <Grid
        item
        xs={12}
        md={isTimeline ? 12 : 5}
        lg={isTimeline ? 12 : 4}
        sx={{ display: 'flex', flexDirection: 'column' }}
      >
        <Tabs
          value={selectedSpanView}
          onChange={(_event, value) => setSelectedSpanView(value as SPAN_VIEW)}
          sx={{ borderBottom: ({ palette }) => `2px solid ${palette.grey[300]}` }}
        >
          {SPAN_VIEWS.map((tab) => (
            <Tab label={tab} value={tab} key={tab} id={tab} />
          ))}
        </Tabs>

        {isTimeline ? (
          <RecordTable selectedNodeId={selectedNodeId} setSelectedNodeId={setSelectedNodeId} root={root} />
        ) : (
          <RecordTree
            selectedNodeId={selectedNodeId}
            setSelectedNodeId={setSelectedNodeId}
            root={root}
            nodeMap={nodeMap}
          />
        )}
      </Grid>

      <Grid item xs={12} md={isTimeline ? 12 : 7} lg={isTimeline ? 12 : 8}>
        <Tabs
          value={selectedTab}
          onChange={(_event, value) => setSelectedTab(value as RECORD_CONTENT_TABS)}
          sx={{ borderBottom: ({ palette }) => `2px solid ${palette.grey[300]}` }}
        >
          {SPAN_TREE_TABS.map((tab) => (
            <Tab label={tab} value={tab} key={tab} id={tab} />
          ))}
          <Box sx={{ flexGrow: 1 }} />
          {GENERAL_TABS.map((tab) => (
            <Tab label={tab} value={tab} key={tab} id={tab} />
          ))}
        </Tabs>

        <Stack gap={2} sx={{ flexGrow: 1, p: 1, mb: 4 }}>
          {getSelectedView()}
        </Stack>
      </Grid>
    </Grid>
  );
}
