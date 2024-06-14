import { Box } from '@mui/material';
import { ReactElement } from 'react';

import StyledTooltip from '@/StyledTooltip';
import { toHumanSpanType } from '@/utils/Span';
import { StackTreeNode } from '@/utils/StackTreeNode';

type SpanTooltipProps = {
  node: StackTreeNode;
  children: ReactElement;
};

export default function SpanTooltip({ node, children }: SpanTooltipProps) {
  const { startTime, endTime, selector, span } = node;

  return (
    <StyledTooltip
      title={
        <Box sx={{ lineHeight: 1.5 }}>
          <span>
            <b>Span type: </b>
            {toHumanSpanType(span?.type)}
          </span>
          <br />
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
