import { AccessTimeRounded } from '@mui/icons-material';
import { Box, SxProps, Theme, Typography } from '@mui/material';
import { TreeItemContentProps, useTreeItemState } from '@mui/x-tree-view/TreeItem';
import clsx from 'clsx';
import { forwardRef } from 'react';

import { SpanTooltip } from '@/SpanTooltip';
import Tag from '@/Tag';
import { StackTreeNode } from '@/utils/StackTreeNode';
import { formatDuration } from '@/utils/utils';
import { SpanTypeTag } from './SpanTypeTag';

type RecordTreeCellProps = TreeItemContentProps & {
  node: StackTreeNode;
};

export const RecordTreeCell = forwardRef(function CustomContent(props: RecordTreeCellProps, ref) {
  const { classes, className, label, itemId, icon: iconProp, expansionIcon, displayIcon, node } = props;

  const { disabled, expanded, selected, focused, handleExpansion, handleSelection } = useTreeItemState(itemId);

  const { timeTaken } = node;

  const icon = iconProp || expansionIcon || displayIcon;

  const handleExpansionClick = (event: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
    handleExpansion(event);
  };

  const handleSelectionClick = (event: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
    handleSelection(event);
  };

  const spanType = node.raw?.['ai.observability.span_type'];

  return (
    <SpanTooltip node={node}>
      <Box
        sx={({ vars }) => ({
          [`&:hover > div`]: {
            background: 'rgba(var(--mui-palette-primary-mainChannel) / var(--mui-palette-action-selectedOpacity))',
          },

          [`&.${classes.selected} > div`]: {
            background: 'rgba(var(--mui-palette-primary-mainChannel) / var(--mui-palette-action-focusOpacity))',
            border: `1px solid ${vars.palette.primary.main}`,
          },
        })}
        className={clsx(className, classes.root, {
          [classes.expanded]: expanded,
          [classes.selected]: selected,
          [classes.focused]: focused,
          [classes.disabled]: disabled,
        })}
        onClick={handleSelectionClick}
        ref={ref as React.Ref<HTMLButtonElement>}
      >
        <Box sx={cellSx}>
          <Box width={icon ? 'calc(100% - 40px)' : '100%'}>
            <Typography sx={ellipsisSx} fontWeight="bold">
              {label}
            </Typography>

            <Box sx={tagsContainerSx}>
              <Tag
                leftIcon={<AccessTimeRounded sx={{ fontSize: 12 }} />}
                sx={tagSx}
                title={formatDuration(timeTaken)}
              />

              <SpanTypeTag spanType={spanType} />
            </Box>
          </Box>

          <Box onClick={(event) => handleExpansionClick(event)}>{icon}</Box>
        </Box>
      </Box>
    </SpanTooltip>
  );
});

const cellSx: SxProps<Theme> = ({ spacing, vars }) => ({
  display: 'flex',
  border: `1px solid ${vars.palette.grey[300]}`,
  p: 1,
  borderRadius: spacing(0.5),
  width: '-webkit-fill-available',
  alignItems: 'center',
  justifyContent: 'space-between',
  '& svg': {
    color: vars.palette.grey[600],
  },
  overflow: 'hidden',
});
const ellipsisSx: SxProps<Theme> = {
  textOverflow: 'ellipsis',
  overflow: 'hidden',
  whiteSpace: 'nowrap',
};

const tagsContainerSx: SxProps<Theme> = { display: 'flex', mt: 0.5, flexWrap: 'wrap', gap: 1 };
const tagSx: SxProps<Theme> = { alignItems: 'center', '& svg': { color: 'grey.900' } };
