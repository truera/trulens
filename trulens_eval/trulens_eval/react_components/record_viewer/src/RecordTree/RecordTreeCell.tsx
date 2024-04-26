import { AccessTimeRounded } from '@mui/icons-material';
import { Box, SxProps, Theme, Typography } from '@mui/material';
import { TreeItemContentProps, useTreeItemState } from '@mui/x-tree-view/TreeItem';
import clsx from 'clsx';
import { forwardRef } from 'react';

import { SpanTooltip } from '@/SpanTooltip';
import Tag from '@/Tag';
import { StackTreeNode } from '@/utils/StackTreeNode';

import SpanIconDisplay from './SpanIconDisplay';

type RecordTreeCellProps = TreeItemContentProps & {
  node: StackTreeNode;
};

export const RecordTreeCell = forwardRef(function CustomContent(props: RecordTreeCellProps, ref) {
  const { classes, className, label, itemId, icon: iconProp, expansionIcon, displayIcon, node } = props;

  const { disabled, expanded, selected, focused, handleExpansion, handleSelection } = useTreeItemState(itemId);

  const { selector, timeTaken } = node;

  const icon = iconProp || expansionIcon || displayIcon;

  const handleExpansionClick = (event: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
    handleExpansion(event);
  };

  const handleSelectionClick = (event: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
    handleSelection(event);
  };

  return (
    <SpanTooltip node={node}>
      <Box
        sx={({ palette }) => ({
          [`&:hover`]: {
            background: `${palette.grey[100]}`,
          },
          [`&.${classes.focused}`]: {
            background: `${palette.grey[50]}`,
          },
          [`&.${classes.focused}:hover`]: {
            background: `${palette.grey[100]}`,
          },

          [`&.${classes.selected}`]: {
            background: `${palette.primary.lighter!}`,
            border: `1px solid ${palette.primary.main}`,
          },
          [`&.${classes.selected}:hover`]: {
            background: `${palette.primary.light}`,
          },
          border: `1px solid ${palette.grey[300]}`,
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
          <SpanIconDisplay spanType={node.span?.type} />

          <Box width={icon ? 'calc(100% - 80px)' : 'calc(100% - 40px)'} sx={{ p: 1, justifyContent: 'space-between' }}>
            <Box>
              <Typography sx={ellipsisSx} fontWeight="bold">
                {label}
              </Typography>
              <Typography variant="code" sx={selectorSx}>
                {selector}
              </Typography>
            </Box>

            <Box sx={tagsContainerSx}>
              <Tag
                leftIcon={<AccessTimeRounded sx={{ fontSize: 12 }} />}
                sx={tagSx}
                severity="info"
                title={`${timeTaken} ms`}
              />
            </Box>
          </Box>

          <Box onClick={(event) => handleExpansionClick(event)} sx={iconContainerSx}>
            {icon}
          </Box>
        </Box>
      </Box>
    </SpanTooltip>
  );
});

const cellSx: SxProps<Theme> = () => ({
  display: 'flex',
  width: '-webkit-fill-available',
  alignItems: 'stretch',
  overflow: 'hidden',
});

const iconContainerSx: SxProps<Theme> = ({ palette }) => ({ pr: 1, color: palette.grey[600] });

const ellipsisSx: SxProps<Theme> = {
  textOverflow: 'ellipsis',
  overflow: 'hidden',
  whiteSpace: 'nowrap',
};

const selectorSx: SxProps<Theme> = {
  textOverflow: 'ellipsis',
  overflow: 'hidden',
  whiteSpace: 'nowrap',
  display: 'inline-block',
  maxWidth: 350,
  wordBreak: 'anywhere',
};

const tagsContainerSx: SxProps<Theme> = { display: 'flex', mt: 0.5, flexWrap: 'wrap' };
const tagSx: SxProps<Theme> = { alignItems: 'center', '& svg': { color: 'grey.900' } };
