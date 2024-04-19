import { ReactElement } from 'react';
import { Box } from '@mui/material';
import { StackTreeNode } from '../utils/StackTreeNode';
import StyledTooltip from '../StyledTooltip/StyledTooltip';

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
