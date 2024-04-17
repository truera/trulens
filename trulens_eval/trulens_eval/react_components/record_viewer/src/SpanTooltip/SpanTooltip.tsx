import { ReactElement } from 'react';
import { Box } from '@mui/material';
import { StackTreeNode } from '../utils/types';
import { getStartAndEndTimesForNode } from '../utils/treeUtils';
import { getSelector } from '../utils/utils';
import StyledTooltip from '../StyledTooltip/StyledTooltip';

type RecordTreeCellTooltipProps = {
  node: StackTreeNode;
  children: ReactElement;
};

export default function RecordTreeCellTooltip({ node, children }: RecordTreeCellTooltipProps) {
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
