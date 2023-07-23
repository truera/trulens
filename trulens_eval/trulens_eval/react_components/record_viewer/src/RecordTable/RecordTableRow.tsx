import { Box, SxProps, TableCell, TableRow, Theme, Tooltip, Typography } from '@mui/material';
import { StackTreeNode } from '../utils/types';
import { getStartAndEndTimesForNode } from '../utils/treeUtils';

type RecordTableProps = {
  nodeWithDepth: { node: StackTreeNode; depth: number };
  totalTime: number;
  treeStart: number;
  selectedNode: string | undefined;
  setSelectedNode: (newNode: string | undefined) => void;
};

function TooltipDescription({ startTime, endTime }: { startTime: number; endTime: number }) {
  return (
    <Box sx={{ fontSize: '0.8rem', lineHeight: 1.5 }}>
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

export default function RecordTableRow({
  nodeWithDepth,
  totalTime,
  treeStart,
  selectedNode,
  setSelectedNode,
}: RecordTableProps) {
  const { node, depth } = nodeWithDepth;
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
        <Box sx={{ ml: depth, display: 'flex', flexDirection: 'column' }}>
          <Typography>{node.name}</Typography>
          <Typography variant="subtitle1">{selector}</Typography>
        </Box>
      </TableCell>
      <TableCell align="right">{timeTaken} ms</TableCell>
      <TableCell sx={{ minWidth: 500 }}>
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
