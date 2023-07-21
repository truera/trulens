import { Box, TableCell, TableRow } from '@mui/material';
import { StackTreeNode } from '../utils/types';
import { getStartAndEndTimesForNode } from '../utils/treeUtils';

type RecordTableProps = { nodeWithDepth: { node: StackTreeNode; depth: number }; totalTime: number; treeStart: number };

export default function RecordTableRow({ nodeWithDepth, totalTime, treeStart }: RecordTableProps) {
  const { node, depth } = nodeWithDepth;
  const { startTime, timeTaken } = getStartAndEndTimesForNode(node);

  return (
    <TableRow>
      <TableCell>
        <Box sx={{ ml: depth }}>{node.name}</Box>
      </TableCell>
      <TableCell align="right">{startTime - treeStart} ms</TableCell>
      <TableCell>
        <Box
          sx={{
            left: `${((startTime - treeStart) / totalTime) * 100}%`,
            width: `${(timeTaken / totalTime) * 100}%`,
          }}
        />
      </TableCell>
    </TableRow>
  );
}
