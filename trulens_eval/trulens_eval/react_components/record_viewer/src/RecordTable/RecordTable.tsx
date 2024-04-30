import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';

import RecordTableRowRecursive from '@/RecordTable/RecordTableRow';
import { StackTreeNode } from '@/utils/StackTreeNode';
import { tableWithoutBorderSx } from '@/utils/styling';

type RecordTableProps = {
  root: StackTreeNode;
  selectedNodeId: string | null;
  setSelectedNodeId: (newId: string | null) => void;
};

export default function RecordTable({ root, selectedNodeId, setSelectedNodeId }: RecordTableProps) {
  const { timeTaken: totalTime, startTime: treeStart } = root;

  return (
    <TableContainer>
      <Table sx={tableWithoutBorderSx} aria-label="Table breakdown of the components in the current app" size="small">
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
