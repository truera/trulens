import { Box } from '@mui/material';
import { ReactElement } from 'react';

import StyledTooltip from '@/StyledTooltip';
import { StackTreeNode } from '@/utils/StackTreeNode';
import { formatTime } from '@/utils/utils';

type SpanTooltipProps = {
  node: StackTreeNode;
  children: ReactElement;
};

export default function SpanTooltip({ node, children }: SpanTooltipProps) {
  const { startTime, endTime, selector } = node;

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
            {formatTime(startTime)}
          </span>
          <br />
          <span>
            <b>End: </b>
            {formatTime(endTime)}
          </span>
        </Box>
      }
    >
      {children}
    </StyledTooltip>
  );
}
