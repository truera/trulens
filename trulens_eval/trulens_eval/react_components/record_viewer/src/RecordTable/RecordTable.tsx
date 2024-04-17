import { Table, TableContainer, TableHead, TableRow, TableCell, TableBody, SxProps, Theme } from '@mui/material';
import { StackTreeNode } from '../utils/types';
import RecordTableRowRecursive from './RecordTableRow';
import { getStartAndEndTimesForNode } from '../utils/treeUtils';

type RecordTableProps = {
  root: StackTreeNode;
  selectedNodeId: string | null;
  setSelectedNodeId: (newId: string | null) => void;
};

export default function RecordTable({ root, selectedNodeId, setSelectedNodeId }: RecordTableProps) {
  const { timeTaken: totalTime, startTime: treeStart } = getStartAndEndTimesForNode(root);

  return (
    <TableContainer>
      <Table sx={recordTableSx} aria-label="Table breakdown of the components in the current app" size="small">
        <TableHead>
          <TableRow>
            <TableCell width={275}>Method</TableCell>
            <TableCell width={75}>Duration</TableCell>
            <TableCell>Timeline</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          <RecordTableRowRecursive
            selectedNodeId={selectedNodeId}
            setSelectedNodeId={setSelectedNodeId}
            node={root}
            depth={0}
            totalTime={totalTime}
            treeStart={treeStart}
          />
        </TableBody>
      </Table>
    </TableContainer>
  );
}

const recordTableSx: SxProps<Theme> = {
  borderRadius: 4,
  border: ({ palette }) => `0.5px solid ${palette.grey[300]}`,
  minWidth: 650,

  '& th': {
    backgroundColor: ({ palette }) => palette.grey[100],
    color: ({ palette }) => palette.grey[600],
    fontWeight: 600,
  },

  '& .MuiTableCell-root': {
    borderRight: ({ palette }) => `1px solid ${palette.grey[300]}`,
  },

  '& .MuiTableCell-root:last-child': {
    borderRight: 'none',
  },

  '& .MuiTableBody-root .MuiTableCell-root': {
    mx: 1,
  },
};
