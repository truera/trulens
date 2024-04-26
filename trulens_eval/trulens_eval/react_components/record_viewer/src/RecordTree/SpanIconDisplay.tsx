import { AccountTreeOutlined, LibraryBooksOutlined, WidgetsOutlined } from '@mui/icons-material';
import { Box, SvgIcon, SxProps, Theme } from '@mui/material';

import { DV_BLUE, DV_GRASS, DV_GREEN, DV_MINT, DV_PLUM, DV_PURPLE, DV_YELLOW, DVColor } from '@/utils/colors';
import { SpanType } from '@/utils/Span';

interface SpanIconProps {
  spanType?: SpanType;
}

const DEFAULT_SPAN_TYPE_PROPS = {
  backgroundColor: DV_BLUE[DVColor.DARK],
  Icon: WidgetsOutlined,
  color: DV_BLUE[DVColor.LIGHT],
};

const SPAN_TYPE_PROPS: { [key in SpanType]: { backgroundColor: string; Icon: typeof SvgIcon; color: string } } = {
  [SpanType.UNTYPED]: DEFAULT_SPAN_TYPE_PROPS,
  [SpanType.ROOT]: {
    backgroundColor: 'primary.main',
    Icon: AccountTreeOutlined,
    color: 'primary.light',
  },
  [SpanType.RETRIEVER]: {
    backgroundColor: DV_YELLOW[DVColor.DARK],
    Icon: LibraryBooksOutlined,
    color: DV_YELLOW[DVColor.LIGHT],
  },
  [SpanType.RERANKER]: {
    backgroundColor: DV_PURPLE[DVColor.DARK],
    Icon: AccountTreeOutlined,
    color: DV_PURPLE[DVColor.LIGHT],
  },
  [SpanType.LLM]: {
    backgroundColor: DV_GREEN[DVColor.DARK],
    Icon: AccountTreeOutlined,
    color: DV_GREEN[DVColor.LIGHT],
  },
  [SpanType.EMBEDDING]: {
    backgroundColor: DV_GRASS[DVColor.DARK],
    Icon: AccountTreeOutlined,
    color: DV_GRASS[DVColor.LIGHT],
  },
  [SpanType.TOOL]: {
    backgroundColor: DV_PLUM[DVColor.DARK],
    Icon: AccountTreeOutlined,
    color: DV_PLUM[DVColor.LIGHT],
  },
  [SpanType.AGENT]: {
    backgroundColor: DV_MINT[DVColor.DARK],
    Icon: AccountTreeOutlined,
    color: DV_MINT[DVColor.LIGHT],
  },
  [SpanType.TASK]: DEFAULT_SPAN_TYPE_PROPS,
  [SpanType.OTHER]: DEFAULT_SPAN_TYPE_PROPS,
};

export default function SpanIconDisplay({ spanType = SpanType.UNTYPED }: SpanIconProps) {
  const { backgroundColor, Icon, color } = SPAN_TYPE_PROPS[spanType] ?? DEFAULT_SPAN_TYPE_PROPS;

  return (
    <Box sx={{ ...containerSx, backgroundColor, color }}>
      <Icon sx={{ ...iconSx, color }} />
    </Box>
  );
}

const iconSx: SxProps<Theme> = {
  p: 1,
};

const containerSx: SxProps<Theme> = {
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'center',
};
