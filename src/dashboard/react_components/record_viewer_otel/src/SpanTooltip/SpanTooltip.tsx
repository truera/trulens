import { Grid2 } from '@mui/material';
import { ReactElement } from 'react';

import StyledTooltip from '@/StyledTooltip';
import { StackTreeNode } from '@/types/StackTreeNode';
import { formatTime } from '@/functions/formatters';

type SpanTooltipProps = {
  node: StackTreeNode;
  children: ReactElement;
};

export default function SpanTooltip({ node, children }: SpanTooltipProps) {
  const { startTime, endTime, name } = node;

  const titleSize = { xs: 2 };
  const valueSize = { xs: 10 };

  return (
    <StyledTooltip
      title={
        <Grid2 container rowSpacing={0.5}>
          <Grid2 size={titleSize}>
            <b>Name: </b>
          </Grid2>
          <Grid2 size={valueSize}>{name}</Grid2>

          <Grid2 size={titleSize}>
            <b>Start: </b>
          </Grid2>
          <Grid2 size={valueSize}>{formatTime(startTime)}</Grid2>

          <Grid2 size={titleSize}>
            <b>End: </b>
          </Grid2>
          <Grid2 size={valueSize}>{formatTime(endTime)}</Grid2>
        </Grid2>
      }
    >
      {children}
    </StyledTooltip>
  );
}
