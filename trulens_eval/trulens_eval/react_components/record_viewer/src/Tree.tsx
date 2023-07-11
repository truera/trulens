import { Streamlit } from 'streamlit-component-lib';
import './RecordViewer.css';
import { getStartAndEndTimesForNode } from './treeUtils';
import { StackTreeNode } from './types';
import { Box, Tooltip } from '@mui/material';

type TreeProps = {
  tree: StackTreeNode;
};

const getChildrenToRender = (tree: StackTreeNode) => {
  const children: { node: StackTreeNode; depth: number }[] = [];
  const { endTime: treeEnd } = getStartAndEndTimesForNode(tree);

  const recursiveGetChildrenToRender = (node: StackTreeNode, depth: number) => {
    const { startTime } = getStartAndEndTimesForNode(node);

    // Ignore calls that happen after the app time. This is indicative of errors.
    if (startTime >= treeEnd) return;

    children.push({ node, depth });

    for (const child of node.children ?? []) {
      recursiveGetChildrenToRender(child, depth + 1);
    }
  };

  recursiveGetChildrenToRender(tree, 0);

  return children;
};

function Node({ node, depth, root }: { node: StackTreeNode; depth: number; root: StackTreeNode }) {
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
          top: depth * 32 + 16,
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

export default function Tree({ tree }: TreeProps) {
  const nodesToRender = getChildrenToRender(tree);

  return (
    <div className="timeline-bar-container">
      {nodesToRender.map(({ node, depth }) => (
        <Node node={node} depth={depth} root={tree} />
      ))}
    </div>
  );
}
