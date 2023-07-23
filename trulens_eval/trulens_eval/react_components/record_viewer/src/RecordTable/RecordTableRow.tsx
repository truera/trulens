import { useEffect } from 'react';
import { ArrowDropDown, ArrowRight } from '@mui/icons-material';
import { Streamlit } from 'streamlit-component-lib';
import { Box, IconButton, SxProps, TableCell, TableRow, Theme, Tooltip, Typography } from '@mui/material';
import { StackTreeNode } from '../utils/types';
import { getStartAndEndTimesForNode } from '../utils/treeUtils';

type RecordTableProps = {
  nodeWithDepth: { node: StackTreeNode; depth: number };
  totalTime: number;
  treeStart: number;
  selectedNode: string | undefined;
  setSelectedNode: (newNode: string | undefined) => void;
  expanded: Set<number | undefined>;
  toggleNodeExpanding: (nodeId: number | undefined) => void;
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

// We can't recursively create the rows because streamlit's frame height update logic
// is still a little wonky :(
export default function RecordTableRow({
  expanded,
  toggleNodeExpanding,
  nodeWithDepth,
  totalTime,
  treeStart,
  selectedNode,
  setSelectedNode,
}: RecordTableProps) {
  const { node, depth } = nodeWithDepth;
  useEffect(() => Streamlit.setFrameHeight());

  if (node.parentNodes.some(({ id }) => !expanded.has(id))) return null;

  const { startTime, timeTaken, endTime } = getStartAndEndTimesForNode(node);

  let selector = 'Select.App';
  if (node.path) selector += `.${node.path}`;

  return (
    <TableRow
      onClick={() => setSelectedNode(node.raw?.perf.start_time ?? undefined)}
      sx={{
        ...recordRowSx,
        background: selectedNode === node.raw?.perf.start_time ? ({ palette }) => palette.primary.lighter : undefined,
      }}
    >
      <TableCell>
        <Box sx={{ ml: depth, display: 'flex', flexDirection: 'row' }}>
          {node.children.length > 0 && (
            <IconButton onClick={() => toggleNodeExpanding(node.id)} disableRipple>
              {expanded.has(node.id) ? <ArrowDropDown /> : <ArrowRight />}
            </IconButton>
          )}
          <Box sx={{ display: 'flex', flexDirection: 'column', ml: node.children.length === 0 ? 5 : 0 }}>
            <Typography>{node.name}</Typography>
            <Typography variant="subtitle1">{selector}</Typography>
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
              ...recordBarSx,
            }}
          />
        </Tooltip>
      </TableCell>
    </TableRow>
  );
}

const recordBarSx: SxProps<Theme> = {
  position: 'relative',
  height: 20,
  background: ({ palette }) => palette.grey[500],
  borderRadius: 0.5,
};

const recordRowSx: SxProps<Theme> = {
  cursor: 'pointer',
  '&:hover': {
    background: ({ palette }) => palette.primary.lighter,
  },
};
