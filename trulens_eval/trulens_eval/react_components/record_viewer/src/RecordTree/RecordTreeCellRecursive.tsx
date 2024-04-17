import { useEffect } from 'react';
import { Streamlit } from 'streamlit-component-lib';
import { SxProps, Theme } from '@mui/material';
import { TreeItem, treeItemClasses } from '@mui/x-tree-view';
import { RecordTreeCell } from './RecordTreeCell';
import { StackTreeNode } from '../utils/types';

type RecordTableRowRecursiveProps = {
  node: StackTreeNode;
  depth: number;
  totalTime: number;
  treeStart: number;
};

export default function RecordTreeCellRecursive({ node, depth, totalTime, treeStart }: RecordTableRowRecursiveProps) {
  useEffect(() => Streamlit.setFrameHeight());

  const isRoot = !node.path;

  const { methodName, name, nodeId } = node;

  const itemLabel = isRoot ? name : [name, methodName].join('.');

  return (
    <TreeItem
      sx={treeItemSx}
      itemId={nodeId}
      label={itemLabel}
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      /* @ts-ignore */
      ContentComponent={RecordTreeCell}
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      /* @ts-ignore */
      ContentProps={{ node }}
    >
      {node.children.map((child) => (
        <RecordTreeCellRecursive
          node={child}
          depth={depth + 1}
          totalTime={totalTime}
          treeStart={treeStart}
          key={`${child.name}-${child.id ?? ''}-${child.endTime ?? ''}`}
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

    [`& .${treeItemClasses.root}:last-of-type > .${treeItemClasses.content}`]: {
      '&::before': {
        width: 0,
      },
    },
  },
});
