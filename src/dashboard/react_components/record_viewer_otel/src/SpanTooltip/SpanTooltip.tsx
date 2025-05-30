import { Grid2 } from '@mui/material';
import { ReactElement } from 'react';

import StyledTooltip from '@/StyledTooltip';
import type { StyledTooltipProps } from '@/StyledTooltip/StyledTooltip';
import { StackTreeNode } from '@/types/StackTreeNode';
import { formatTime } from '@/functions/formatters';
import { getNodeSpanType } from '@/functions/getNodeSpanType';
import { ORPHANED_NODES_PARENT_ID } from '@/constants/node';

type SpanTooltipProps = {
  node: StackTreeNode;
  children: ReactElement;
  placement?: StyledTooltipProps['placement'];
};

export default function SpanTooltip({ node, children, placement }: SpanTooltipProps) {
  const { startTime, endTime, name, id } = node;

  if (id === ORPHANED_NODES_PARENT_ID) {
    return (
      <StyledTooltip
        placement={placement}
        title={
          'Nodes that are nested under this node belong to the record, but we were unable to determine their relationship to the rest of the nodes. This could be due to missing spans or incorrect span ids.'
        }
      >
        {children}
      </StyledTooltip>
    );
  }

  const titleSize = { xs: 3 };
  const valueSize = { xs: 9 };
  const spanType = getNodeSpanType(node);

  return (
    <StyledTooltip
      placement={placement}
      title={
        <Grid2 container rowSpacing={0.5}>
          <Grid2 size={titleSize}>
            <b>Name: </b>
          </Grid2>
          <Grid2 size={valueSize}>{name}</Grid2>

          {spanType !== 'Unknown' && (
            <>
              <Grid2 size={titleSize}>
                <b>Span type: </b>
              </Grid2>
              <Grid2 size={valueSize}>{spanType}</Grid2>
            </>
          )}

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
