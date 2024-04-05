import { useEffect } from 'react';
import { Streamlit } from 'streamlit-component-lib';
import { Box, SxProps, Theme } from '@mui/material';
import { TreeItem, treeItemClasses } from '@mui/x-tree-view';
import { StackTreeNode } from '../utils/types';
import { CustomContent } from './FolderTreeView';

type RecordTableRowRecursiveProps = {
  node: StackTreeNode;
  depth: number;
  totalTime: number;
  treeStart: number;
  selectedNode: string | undefined;
  setSelectedNode: (newNode: string | undefined) => void;
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

export default function RecordTreeCellRecursive({
  node,
  depth,
  totalTime,
  treeStart,
  selectedNode,
  setSelectedNode,
}: RecordTableRowRecursiveProps) {
  useEffect(() => Streamlit.setFrameHeight());

  let selector = 'Select.App';

  if (node.path) selector += `.${node.path}`;

  const isRoot = !node.path;
  const nodeStartTime = isRoot ? '' : node.raw?.perf.start_time;

  const { id, methodName, name } = node;

  const isNodeSelected = selectedNode === nodeStartTime;

  const itemId = [id, methodName, name].join('-');
  const itemLabel = [name, methodName].join('.');

  return (
    <TreeItem
      sx={treeItemSx}
      itemId={itemId}
      label={itemLabel}
      ContentComponent={CustomContent}
      ContentProps={{ node }}
    >
      {node.children.map((child) => (
        <RecordTreeCellRecursive
          selectedNode={selectedNode}
          setSelectedNode={setSelectedNode}
          node={child}
          depth={depth + 1}
          totalTime={totalTime}
          treeStart={treeStart}
          key={`${child.name}-${child.id ?? ''}-${child.endTime?.toISOString() ?? ''}`}
        />
      ))}
    </TreeItem>
  );
}

const treeItemSx: SxProps<Theme> = ({ spacing, palette }) => ({
  [`& .${treeItemClasses.content}`]: {
    textAlign: 'left',
    position: 'relative',
    zIndex: 1,
    p: 0,
  },
  [`& .${treeItemClasses.content} ${treeItemClasses.label}`]: {
    paddingLeft: spacing(1),
  },
  [`& .${treeItemClasses.root}`]: {
    position: 'relative',
    // Final vertical segment - achieving the curve effect.
    '&:last-of-type': {
      '&::before': {
        // Magic value, based on the height of a single cell.
        height: `calc(54px + ${spacing(1)})`,
        width: spacing(3),
        borderBottom: `1px solid ${palette.grey[300]}`,
      },
    },
    '&::before': {
      content: '""',
      display: 'block',
      position: 'absolute',
      height: `calc(100% + ${spacing(3)})`,
      borderBottomLeftRadius: 4,
      borderLeft: `1px solid ${palette.grey[300]}`,

      // Magic values, based on paddings
      left: spacing(-3),
      top: spacing(-1),
    },
  },
  [`& .${treeItemClasses.groupTransition}`]: {
    marginLeft: 0,
    paddingLeft: spacing(3),
    [`& .${treeItemClasses.root}`]: {
      pt: 1,
    },

    [`& .${treeItemClasses.root}  .${treeItemClasses.content}`]: {
      '&::before': {
        content: '""',
        position: 'absolute',
        display: 'block',
        width: spacing(3),
        height: spacing(1),
        top: '50%',
        borderBottom: `1px solid ${palette.grey[300]}`,
        transform: 'translate(-100%, -50%)',
      },
    },

    '& .MuiTreeItem-root:last-of-type > .MuiTreeItem-content': {
      '&::before': {
        width: 0,
      },
    },
  },
});
