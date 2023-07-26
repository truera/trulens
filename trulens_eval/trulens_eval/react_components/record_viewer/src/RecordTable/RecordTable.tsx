import { useEffect, useState } from 'react';
import { Streamlit } from 'streamlit-component-lib';
import { Table, TableContainer, TableHead, TableRow, TableCell, TableBody, SxProps, Theme } from '@mui/material';
import { StackTreeNode } from '../utils/types';
import RecordTableRowRecursive from './RecordTableRow';
import { getStartAndEndTimesForNode } from '../utils/treeUtils';

type RecordTableProps = {
  root: StackTreeNode;
};

export default function RecordTable({ root }: RecordTableProps) {
  const [selectedNode, setSelectedNode] = useState<string>();

  useEffect(() => Streamlit.setComponentValue(selectedNode), [selectedNode]);

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
            selectedNode={selectedNode}
            setSelectedNode={setSelectedNode}
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
  border: ({ palette }) => `1px solid ${palette.primary.light}`,
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
