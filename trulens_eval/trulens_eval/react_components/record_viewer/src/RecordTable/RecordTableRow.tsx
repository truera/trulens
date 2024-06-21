import { ArrowDropDown, ArrowRight } from '@mui/icons-material';
import { Box, IconButton, SxProps, TableCell, TableRow, Theme, Typography } from '@mui/material';
import { useEffect, useState } from 'react';
import { Streamlit } from 'streamlit-component-lib';

import { SpanTooltip } from '@/SpanTooltip';
import { StackTreeNode } from '@/utils/StackTreeNode';
import { formatDuration } from '@/utils/utils';

type RecordTableRowRecursiveProps = {
  node: StackTreeNode;
  depth: number;
  totalTime: number;
  treeStart: number;
  selectedNodeId: string | null;
  setSelectedNodeId: (newNode: string | null) => void;
};

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

  const { nodeId, startTime, timeTaken, selector, label } = node;

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
              <Typography fontWeight="bold">{label}</Typography>
              <Typography variant="code" sx={{ ml: 1, px: 1 }}>
                {selector}
              </Typography>
            </Box>
          </Box>
        </TableCell>
        <TableCell align="right">{formatDuration(timeTaken)}</TableCell>
        <TableCell sx={{ minWidth: 500, padding: 0 }}>
          <SpanTooltip node={node}>
            <Box
              sx={{
                left: `${((startTime - treeStart) / totalTime) * 100}%`,
                width: `${(timeTaken / totalTime) * 100}%`,
                background: ({ palette }) =>
                  selectedNodeId === null || isNodeSelected ? palette.grey[500] : palette.grey[300],
                ...recordBarSx,
              }}
            />
          </SpanTooltip>
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
              key={child.nodeId}
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
