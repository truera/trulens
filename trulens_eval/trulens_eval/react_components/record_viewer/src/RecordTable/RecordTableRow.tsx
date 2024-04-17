import { useEffect, useState } from 'react';
import { ArrowDropDown, ArrowRight } from '@mui/icons-material';
import { Streamlit } from 'streamlit-component-lib';
import { Box, IconButton, SxProps, TableCell, TableRow, Theme, Tooltip, Typography } from '@mui/material';
import { StackTreeNode } from '../utils/types';
import { getStartAndEndTimesForNode } from '../utils/treeUtils';
import { getSelector } from '../utils/utils';

type RecordTableRowRecursiveProps = {
  node: StackTreeNode;
  depth: number;
  totalTime: number;
  treeStart: number;
  selectedNodeId: string | null;
  setSelectedNodeId: (newNode: string | null) => void;
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
  selectedNodeId,
  setSelectedNodeId,
}: RecordTableRowRecursiveProps) {
  useEffect(() => Streamlit.setFrameHeight());

  const [expanded, setExpanded] = useState<boolean>(true);

  const { name, methodName, nodeId } = node;
  const { startTime, timeTaken, endTime } = getStartAndEndTimesForNode(node);
  const selector = getSelector(node);

  const isRoot = !node.path;
  const itemLabel = isRoot ? name : [name, methodName].join('.');

  const isNodeSelected = selectedNodeId === nodeId;

  return (
    <>
      <TableRow
        onClick={() => setSelectedNodeId(nodeId ?? null)}
        sx={{
          ...recordRowSx,
          background: isNodeSelected ? ({ palette }) => palette.primary.lighter : undefined,
        }}
      >
        <TableCell>
          <Box sx={{ ml: depth, display: 'flex', flexDirection: 'row' }}>
            {node.children.length > 0 && (
              <IconButton onClick={() => setExpanded(!expanded)} disableRipple size="small">
                {expanded ? <ArrowDropDown /> : <ArrowRight />}
              </IconButton>
            )}
            <Box sx={{ display: 'flex', alignItems: 'center', ml: node.children.length === 0 ? 5 : 0 }}>
              <Typography>{itemLabel}</Typography>
              <Typography variant="code" sx={{ ml: 1, px: 1 }}>
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
                  selectedNodeId === null || isNodeSelected ? palette.grey[500] : palette.grey[300],
                ...recordBarSx,
              }}
            />
          </Tooltip>
        </TableCell>
      </TableRow>
      {expanded
        ? node.children.map((child) => (
            <RecordTableRowRecursive
              selectedNodeId={selectedNodeId}
              setSelectedNodeId={setSelectedNodeId}
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
