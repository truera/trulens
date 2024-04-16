import { forwardRef, ReactElement } from 'react';
import clsx from 'clsx';
import { AccessTimeRounded } from '@mui/icons-material';
import { Box, SxProps, Theme, Typography } from '@mui/material';
import { useTreeItemState, TreeItemContentProps } from '@mui/x-tree-view/TreeItem';
import { StackTreeNode } from '../utils/types';
import { getStartAndEndTimesForNode } from '../utils/treeUtils';
import Tag from '../Tag/Tag';
import { getSelector } from '../utils/utils';
import StyledTooltip from '../StyledTooltip/StyledTooltip';

type RecordTreeCellTooltipProps = {
  node: StackTreeNode;
  children: ReactElement;
};
function RecordTreeCellTooltip({ node, children }: RecordTreeCellTooltipProps) {
  const { startTime, endTime } = getStartAndEndTimesForNode(node);
  const selector = getSelector(node);
  return (
    <StyledTooltip
      title={
        <Box sx={{ lineHeight: 1.5 }}>
          <span>
            <b>Selector: </b>
            {selector}
          </span>
          <br />
          <span>
            <b>Start: </b>
            {new Date(startTime).toLocaleDateString()} {new Date(startTime).toLocaleTimeString()}
          </span>
          <br />
          <span>
            <b>End: </b>
            {new Date(endTime).toLocaleDateString()} {new Date(endTime).toLocaleTimeString()}
          </span>
        </Box>
      }
    >
      {children}
    </StyledTooltip>
  );
}

type RecordTreeCellProps = TreeItemContentProps & {
  node: StackTreeNode;
};

export const RecordTreeCell = forwardRef(function CustomContent(props: RecordTreeCellProps, ref) {
  const { classes, className, label, itemId, icon: iconProp, expansionIcon, displayIcon, node } = props;

  const { disabled, expanded, selected, focused, handleExpansion, handleSelection } = useTreeItemState(itemId);
  const selector = getSelector(node);

  const { timeTaken } = getStartAndEndTimesForNode(node);

  const icon = iconProp || expansionIcon || displayIcon;

  const handleExpansionClick = (event: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
    handleExpansion(event);
  };

  const handleSelectionClick = (event: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
    handleSelection(event);
  };
  return (
    <RecordTreeCellTooltip node={node}>
      <Box
        sx={({ palette }) => ({
          [`&:hover > div`]: {
            background: `${palette.grey[100]}`,
          },
          [`&.${classes.focused} > div`]: {
            background: `${palette.grey[50]}`,
          },
          [`&.${classes.focused}:hover > div`]: {
            background: `${palette.grey[100]}`,
          },

          [`&.${classes.selected} > div`]: {
            background: `${palette.primary.lighter!}`,
            border: `1px solid ${palette.primary.main}`,
          },
          [`&.${classes.selected}:hover > div`]: {
            background: `${palette.primary.light}`,
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
            <Typography color="grey.600" sx={ellipsisSx}>
              {selector}
            </Typography>

            <Box sx={tagsContainerSx}>
              <Tag
                leftIcon={<AccessTimeRounded sx={{ fontSize: 12 }} />}
                sx={tagSx}
                severity="info"
                title={`${timeTaken} ms`}
              />
            </Box>
          </Box>

          <Box onClick={(event) => handleExpansionClick(event)}>{icon}</Box>
        </Box>
      </Box>
    </RecordTreeCellTooltip>
  );
});

const cellSx: SxProps<Theme> = ({ spacing, palette }) => ({
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
  overflow: 'hidden',
});
const ellipsisSx: SxProps<Theme> = {
  textOverflow: 'ellipsis',
  overflow: 'hidden',
  width: '100%',
  whiteSpace: 'nowrap',
};

const tagsContainerSx: SxProps<Theme> = { display: 'flex', mt: 0.5, flexWrap: 'wrap' };
const tagSx: SxProps<Theme> = { alignItems: 'center', '& svg': { color: 'grey.900' } };
