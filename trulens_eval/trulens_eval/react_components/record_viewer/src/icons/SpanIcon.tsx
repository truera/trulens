import {
  AccountTreeOutlined,
  AutoAwesomeOutlined,
  BatchPredictionOutlined,
  BuildOutlined,
  LeaderboardOutlined,
  LibraryBooksOutlined,
  PolylineOutlined,
  SupportAgentOutlined,
  WidgetsOutlined,
} from '@mui/icons-material';
import { SvgIcon } from '@mui/material';

import { DV_BLUE, DV_GRASS, DV_LAVENDER, DV_MINT, DV_PLUM, DV_PURPLE, DV_YELLOW, DVColor } from '@/utils/colors';
import { SpanType } from '@/utils/Span';

export const DEFAULT_SPAN_TYPE_PROPS = {
  backgroundColor: DV_BLUE[DVColor.DARK],
  Icon: WidgetsOutlined,
  color: DV_BLUE[DVColor.LIGHT],
};

export const SPAN_TYPE_PROPS: { [key in SpanType]: { backgroundColor: string; Icon: typeof SvgIcon; color: string } } =
  {
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
      Icon: LeaderboardOutlined,
      color: DV_PURPLE[DVColor.LIGHT],
    },
    [SpanType.LLM]: {
      backgroundColor: DV_YELLOW[DVColor.SECONDARY_LIGHT],
      Icon: AutoAwesomeOutlined,
      color: DV_YELLOW[DVColor.DARK],
    },
    [SpanType.EMBEDDING]: {
      backgroundColor: DV_GRASS[DVColor.DARK],
      Icon: PolylineOutlined,
      color: DV_GRASS[DVColor.LIGHT],
    },
    [SpanType.TOOL]: {
      backgroundColor: DV_PLUM[DVColor.DARK],
      Icon: BuildOutlined,
      color: DV_PLUM[DVColor.LIGHT],
    },
    [SpanType.AGENT]: {
      backgroundColor: DV_MINT[DVColor.DARK],
      Icon: SupportAgentOutlined,
      color: DV_MINT[DVColor.LIGHT],
    },
    [SpanType.MEMORY]: {
      backgroundColor: DV_LAVENDER[DVColor.DARK],
      Icon: BatchPredictionOutlined,
      color: DV_LAVENDER[DVColor.LIGHT],
    },
    [SpanType.TASK]: DEFAULT_SPAN_TYPE_PROPS,
    [SpanType.OTHER]: DEFAULT_SPAN_TYPE_PROPS,
  };
