/* eslint-disable jsx-a11y/no-static-element-interactions */
/* eslint-disable jsx-a11y/click-events-have-key-events */
import * as React from 'react';
import clsx from 'clsx';
import { AccessTimeRounded } from '@mui/icons-material';
import Box from '@mui/material/Box';
import { useTreeItemState, TreeItemContentProps } from '@mui/x-tree-view/TreeItem';
import Typography from '@mui/material/Typography';
import { StackTreeNode } from '../utils/types';
import { getStartAndEndTimesForNode } from '../utils/treeUtils';
import Tag from '../Tag/Tag';

type CellProps = TreeItemContentProps & {
  node: StackTreeNode;
};

export const CustomContent = React.forwardRef(function CustomContent(props: CellProps, ref) {
  const { classes, className, label, itemId, icon: iconProp, expansionIcon, displayIcon, node } = props;

  const { disabled, expanded, focused, handleExpansion, handleSelection, preventSelection } = useTreeItemState(itemId);

  let selector = 'Select.App';

  if (node.path) selector += `.${node.path}`;
  const { startTime, timeTaken, endTime } = getStartAndEndTimesForNode(node);

  const icon = iconProp || expansionIcon || displayIcon;

  const handleMouseDown = (event: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
    preventSelection(event);
  };

  const handleExpansionClick = (event: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
    handleExpansion(event);
  };

  const handleSelectionClick = (event: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
    handleSelection(event);
  };
  return (
    <Box
      sx={({ palette }) => ({
        [`&:hover > div`]: {
          background: `${palette.grey[100]}`,
        },
        [`&.${classes.focused} > div`]: {
          background: `${palette.primary.lighter!}`,
          border: `1px solid ${palette.primary.main}`,
        },
        [`&.${classes.focused}:hover > div`]: {
          background: `${palette.primary.light}`,
        },
      })}
      className={clsx(className, classes.root, {
        [classes.expanded]: expanded,
        [classes.selected]: focused, // TODO: fixme
        [classes.focused]: focused,
        [classes.disabled]: disabled,
      })}
      onClick={handleExpansionClick}
      onMouseDown={handleMouseDown}
      ref={ref as React.Ref<HTMLButtonElement>}
    >
      <Box
        sx={({ spacing, palette }) => ({
          display: 'flex',
          border: `1px solid ${palette.grey[300]}`,
          p: 1,
          borderRadius: spacing(0.5),
          width: '100%',
          alignItems: 'center',
          justifyContent: 'space-between',
          '& svg': {
            color: palette.grey[600],
          },
        })}
      >
        <Box>
          <Typography
            onClick={handleSelectionClick}
            sx={{
              color: 'text.primary',
              fontWeight: ({ typography }) => typography.fontWeightBold,
            }}
          >
            {label}
          </Typography>
          <Typography
            onClick={handleSelectionClick}
            sx={{
              color: 'grey.600',
            }}
          >
            {selector}
          </Typography>

          <Box sx={{ display: 'flex', mt: 0.5, flexWrap: 'wrap' }}>
            <Tag leftIcon={<AccessTimeRounded fontSize="small" />} severity="info" title={`${timeTaken} ms`} />
          </Box>
        </Box>

        {icon}
      </Box>
    </Box>
  );
});
