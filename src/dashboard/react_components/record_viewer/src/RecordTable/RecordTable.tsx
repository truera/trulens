import { SxProps, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Theme } from '@mui/material';

import RecordTableRowRecursive from '@/RecordTable/RecordTableRow';
import { StackTreeNode } from '@/utils/StackTreeNode';

type RecordTableProps = {
  root: StackTreeNode;
  selectedNodeId: string | null;
  setSelectedNodeId: (newId: string | null) => void;
};

export default function RecordTable({ root, selectedNodeId, setSelectedNodeId }: RecordTableProps) {
  const { timeTaken: totalTime, startTime: treeStart } = root;

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
  border: ({ vars }) => `0.5px solid ${vars.palette.grey[300]}`,
  minWidth: 650,

  '& th': {
    backgroundColor: ({ vars }) => vars.palette.grey[100],
    color: ({ vars }) => vars.palette.grey[600],
    fontWeight: 600,
  },

  '& .MuiTableCell-root': {
    borderRight: ({ vars }) => `1px solid ${vars.palette.grey[300]}`,
  },

  '& .MuiTableCell-root:last-child': {
    borderRight: 'none',
  },

  '& .MuiTableBody-root .MuiTableCell-root': {
    mx: 1,
  },
};
