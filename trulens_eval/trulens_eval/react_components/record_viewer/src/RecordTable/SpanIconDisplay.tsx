import { Box, SxProps, Theme } from '@mui/material';

import { DEFAULT_SPAN_TYPE_PROPS, SPAN_TYPE_PROPS } from '@/icons/SpanIcon';
import { SpanType } from '@/utils/Span';

interface SpanIconProps {
  spanType?: SpanType;
}

export default function SpanIconDisplay({ spanType = SpanType.UNTYPED }: SpanIconProps) {
  const { backgroundColor, Icon, color } = SPAN_TYPE_PROPS[spanType] ?? DEFAULT_SPAN_TYPE_PROPS;

  return (
    <Box sx={{ ...containerSx, backgroundColor, color }}>
      <Icon sx={{ ...iconSx, color }} />
    </Box>
  );
}

const iconSx: SxProps<Theme> = {
  p: 0.5,
  height: '16px',
  width: '16px',
};

const containerSx: SxProps<Theme> = {
  display: 'flex',
  borderRadius: ({ spacing }) => spacing(0.5),
  flexDirection: 'column',
  justifyContent: 'center',
};
