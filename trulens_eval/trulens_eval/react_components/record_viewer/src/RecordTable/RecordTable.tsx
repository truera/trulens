import { Paper, Table, TableContainer, TableHead, TableRow, TableCell, TableBody, SxProps, Theme } from '@mui/material';
import { StackTreeNode } from '../utils/types';
import RecordTableRow from './RecordTableRow';
import { getNodesToRender, getStartAndEndTimesForNode } from '../utils/treeUtils';

type RecordTableProps = {
  root: StackTreeNode;
};

export default function RecordTable({ root }: RecordTableProps) {
  const nodesToRender = getNodesToRender(root);
  const { timeTaken: totalTime, startTime: treeStart } = getStartAndEndTimesForNode(root);

  return (
    <TableContainer component={Paper}>
      <Table sx={recordTableSx} aria-label="Table breakdown of the components in the current app" size="small">
        <TableHead>
          <TableRow>
            <TableCell width={275}>Method</TableCell>
            <TableCell width={75}>Duration</TableCell>
            <TableCell>Timeline</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {nodesToRender.map((nodeWithDepth) => (
            <RecordTableRow nodeWithDepth={nodeWithDepth} totalTime={totalTime} treeStart={treeStart} />
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

const recordTableSx: SxProps<Theme> = {
  borderRadius: 4,
  border: ({ palette }) => `1px solid ${palette.primary.light}`,
  minWidth: 650,

  '& th': {
    backgroundColor: ({ palette }) => palette.grey[100],
    color: ({ palette }) => palette.grey[600],
    fontWeight: 600,
  },
};
