import { useEffect, useState } from 'react';
import { ArrowDropDown, ArrowRight } from '@mui/icons-material';
import { Streamlit } from 'streamlit-component-lib';
import { Box, IconButton, SxProps, TableCell, TableRow, Theme, Tooltip, Typography } from '@mui/material';
import { StackTreeNode } from '../utils/types';
import { getStartAndEndTimesForNode } from '../utils/treeUtils';

type RecordTableRowRecursiveProps = {
  node: StackTreeNode;
  depth: number;
  totalTime: number;
  treeStart: number;
  selectedNode: string | undefined;
  setSelectedNode: (newNode: string | undefined) => void;
};

function TooltipDescription({ startTime, endTime }: { startTime: number; endTime: number }) {
  return (
    <Box sx={{ lineHeight: 1.5 }}>
      <span>
        <b>Start: </b>
        {new Date(startTime).toISOString()}
      </span>
      <br />
      <span>
        <b>End: </b>
        {new Date(endTime).toISOString()}
      </span>
    </Box>
  );
}

export default function RecordTableRowRecursive({
  node,
  depth,
  totalTime,
  treeStart,
  selectedNode,
  setSelectedNode,
}: RecordTableRowRecursiveProps) {
  useEffect(() => Streamlit.setFrameHeight());

  const [expanded, setExpanded] = useState<boolean>(true);

  const { startTime, timeTaken, endTime } = getStartAndEndTimesForNode(node);

  let selector = 'Select.App';
  if (node.path) selector += `${node.path}`;

  const isNodeSelected = selectedNode === node.raw?.perf.start_time;

  return (
    <>
      <TableRow
        onClick={() => setSelectedNode(node.raw?.perf.start_time ?? undefined)}
        sx={{
          ...recordRowSx,
          background: isNodeSelected ? ({ palette }) => palette.primary.lighter : undefined,
        }}
      >
        <TableCell>
          <Box sx={{ ml: depth, display: 'flex', flexDirection: 'row' }}>
            {node.children.length > 0 && (
              <IconButton onClick={() => setExpanded(!expanded)} disableRipple>
                {expanded ? <ArrowDropDown /> : <ArrowRight />}
              </IconButton>
            )}
            <Box sx={{ display: 'flex', flexDirection: 'column', ml: node.children.length === 0 ? 5 : 0 }}>
              <Typography>
                {node.name}
                {node.methodName ? `.${node.methodName}` : ''}
              </Typography>
              <Typography variant="code" sx={{ px: 0.5 }}>
                {selector}
              </Typography>
            </Box>
          </Box>
        </TableCell>
        <TableCell align="right">{timeTaken} ms</TableCell>
        <TableCell sx={{ minWidth: 500, padding: 0 }}>
          <Tooltip title={<TooltipDescription startTime={startTime} endTime={endTime} />}>
            <Box
              sx={{
                left: `${((startTime - treeStart) / totalTime) * 100}%`,
                width: `${(timeTaken / totalTime) * 100}%`,
                background: ({ palette }) =>
                  selectedNode === undefined || isNodeSelected ? palette.grey[500] : palette.grey[300],
                ...recordBarSx,
              }}
            />
          </Tooltip>
        </TableCell>
      </TableRow>
      {expanded
        ? node.children.map((child) => (
            <RecordTableRowRecursive
              selectedNode={selectedNode}
              setSelectedNode={setSelectedNode}
              node={child}
              depth={depth + 1}
              totalTime={totalTime}
              treeStart={treeStart}
              key={`${child.name}-${child.id ?? ''}-${child.endTime?.toISOString() ?? ''}`}
            />
          ))
        : null}
    </>
  );
}

const recordBarSx: SxProps<Theme> = {
  position: 'relative',
  height: 20,
  borderRadius: 0.5,
};

const recordRowSx: SxProps<Theme> = {
  cursor: 'pointer',
  '&:hover': {
    background: ({ palette }) => palette.primary.lighter,
  },
};
