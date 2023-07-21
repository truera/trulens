import { Box, Tooltip } from '@mui/material';
import { Streamlit } from 'streamlit-component-lib';
import './RecordViewer.css';
import { getNodesToRender, getStartAndEndTimesForNode } from './utils/treeUtils';
import { StackTreeNode } from './utils/types';
import { TIME_DISPLAY_HEIGHT_BUFFER } from './utils/styling';

// TODO: fix in later release
/* eslint-disable jsx-a11y/click-events-have-key-events,jsx-a11y/no-static-element-interactions */

export const BAR_HEIGHT = 32;

type TreeProps = {
  root: StackTreeNode;
};

function NodeBar({ node, depth, root }: { node: StackTreeNode; depth: number; root: StackTreeNode }) {
  const { startTime, timeTaken } = getStartAndEndTimesForNode(node);
  const { timeTaken: totalTime, startTime: treeStart } = getStartAndEndTimesForNode(root);

  const { name, methodName, path } = node;

  const description = (
    <Box className="description">
      <b>{name}</b>
      <span>
        <b>Time taken:</b> {timeTaken}ms
      </span>
      {methodName && (
        <span>
          <b>Method name:</b> {methodName}
        </span>
      )}
      {path && (
        <span>
          <b>Path:</b> {path}
        </span>
      )}
    </Box>
  );

  return (
    <Tooltip title={description} arrow>
      <div
        className="timeline"
        style={{
          left: `${((startTime - treeStart) / totalTime) * 100}%`,
          width: `${(timeTaken / totalTime) * 100}%`,
          top: depth * BAR_HEIGHT + TIME_DISPLAY_HEIGHT_BUFFER,
        }}
        onClick={() => {
          Streamlit.setComponentValue(node.raw?.perf.start_time ?? null);
        }}
      >
        <span className="timeline-component-name">{node.name}</span>
        <span className="timeline-time-taken">{timeTaken}ms</span>
      </div>
    </Tooltip>
  );
}

export default function TimelineBars({ root: tree }: TreeProps) {
  const nodesToRender = getNodesToRender(tree);

  return (
    <div className="timeline-bar-container">
      <div style={{ position: 'relative' }}>
        {nodesToRender.map(({ node, depth }) => (
          <NodeBar node={node} depth={depth} root={tree} />
        ))}
      </div>
    </div>
  );
}
