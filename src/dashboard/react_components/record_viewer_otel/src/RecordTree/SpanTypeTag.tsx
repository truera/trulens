import { getSpanTypeTitle } from '@/functions/getSpanTypeTitle';
import Tag from '@/Tag';
import { SxProps, Theme } from '@mui/material';

export interface SpanTypeTagProps {
  spanType: string;
  sx?: SxProps<Theme>;
}

export const SpanTypeTag = (props: SpanTypeTagProps) => {
  const { spanType, sx } = props;

  if (spanType === 'unknown') return null;

  return <Tag sx={sx} title={getSpanTypeTitle(spanType)} />;
};
