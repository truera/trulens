import { ArrowDropDown, ArrowRight } from '@mui/icons-material';
import { Box, IconButton, SxProps, TableCell, TableRow, Theme, Typography } from '@mui/material';
import { useEffect, useState } from 'react';
import { Streamlit } from 'streamlit-component-lib';

import { SpanTooltip } from '@/SpanTooltip';
import { StackTreeNode } from '@/types/StackTreeNode';
import { formatDuration } from '@/functions/formatters';
import { getNodeSpanType } from '@/functions/getNodeSpanType';
import { ORPHANED_NODES_PARENT_ID } from '@/constants/node';
import { combineSx } from '@/utils/styling';

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

  const { id, startTime, timeTaken, label } = node;

  const isNodeSelected = selectedNodeId === id;
  const isOrphanContainer = id === ORPHANED_NODES_PARENT_ID;
  const spanType = getNodeSpanType(node);

  const handleRowClick = () => {
    if (isOrphanContainer) return;
    setSelectedNodeId(id ?? null);
  };

  return (
    <>
      <SpanTooltip node={node} placement="bottom-start">
        <TableRow
          onClick={handleRowClick}
          sx={combineSx(recordRowSx, isOrphanContainer ? orphanContainerRowSx : {}, {
            background: isNodeSelected
              ? 'rgba(var(--mui-palette-primary-mainChannel) / calc(var(--mui-palette-action-focusOpacity)))'
              : undefined,
          })}
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
              </Box>
            </Box>
          </TableCell>
          <TableCell align="right">{isOrphanContainer ? '-' : formatDuration(timeTaken)}</TableCell>
          <TableCell>{spanType === 'Unknown' ? '-' : spanType}</TableCell>
          <TableCell sx={{ minWidth: 500, padding: 0 }}>
            <SpanTooltip node={node}>
              <Box
                sx={{
                  left: `${((startTime - treeStart) / totalTime) * 100}%`,
                  width: `${(timeTaken / totalTime) * 100}%`,
                  background: ({ vars }) =>
                    selectedNodeId === null || isNodeSelected ? vars.palette.grey[500] : vars.palette.grey[300],
                  ...recordBarSx,
                }}
              />
            </SpanTooltip>
          </TableCell>
        </TableRow>
      </SpanTooltip>

      {expanded
        ? node.children.map((child) => (
            <RecordTableRowRecursive
              selectedNodeId={selectedNodeId}
              setSelectedNodeId={setSelectedNodeId}
              node={child}
              depth={depth + 1}
              totalTime={totalTime}
              treeStart={treeStart}
              key={child.id}
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
    background: 'rgba(var(--mui-palette-primary-mainChannel) / var(--mui-palette-action-selectedOpacity))',
  },
};

const orphanContainerRowSx: SxProps<Theme> = {
  background: 'rgba(var(--mui-palette-error-mainChannel) / var(--mui-palette-action-disabledOpacity))',
  '&:hover': {
    background: 'rgba(var(--mui-palette-error-mainChannel) / var(--mui-palette-action-selectedOpacity))',
  },
};
