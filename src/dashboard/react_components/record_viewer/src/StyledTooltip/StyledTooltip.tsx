import Tooltip, { TooltipProps } from '@mui/material/Tooltip';
import { ReactElement, ReactNode } from 'react';

export default function StyledTooltip({
  title,
  placement,
  children,
}: {
  title: ReactNode;
  placement?: TooltipProps['placement'];
  children: ReactElement;
}) {
  return (
    <Tooltip
      title={title}
      slotProps={{
        tooltip: {
          sx: {
            // The typing for tooltip is weird and doesn't understand SxProps<Theme>
            // properly.
            /* eslint-disable */
            color: ({ vars }) => vars.palette.text.primary,
            backgroundColor: ({ vars }) => vars.palette.grey[100],
            border: ({ palette }) => `1px solid ${palette.grey[300]}`,
            boxShadow: `0px 8px 16px 0px #0000000D`,
            /* eslint-enable */
          },
        },
      }}
      placement={placement}
    >
      {children}
    </Tooltip>
  );
}
