import Tooltip, { TooltipProps } from '@mui/material/Tooltip';
import { ReactElement, ReactNode } from 'react';

export interface StyledTooltipProps {
  title: ReactNode;
  placement?: TooltipProps['placement'];
  children: ReactElement;
}

export default function StyledTooltip({ title, placement, children }: StyledTooltipProps) {
  return (
    <Tooltip
      title={title}
      slotProps={{
        tooltip: {
          sx: {
            color: ({ palette }) => palette.text.primary,
            backgroundColor: ({ palette }) => palette.grey[100],
            border: ({ palette }) => `1px solid ${palette.grey[300]}`,
            boxShadow: `0px 8px 16px 0px #0000000D`,
          },
        },
      }}
      placement={placement}
    >
      {children}
    </Tooltip>
  );
}
