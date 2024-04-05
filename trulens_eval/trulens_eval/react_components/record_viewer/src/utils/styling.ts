import { SxProps, Theme } from '@mui/material';

// eslint-disable-next-line import/prefer-default-export
export const TIME_DISPLAY_HEIGHT_BUFFER = 16;

/**
 * Utility function to combine sx props. See
 * https://mui.com/system/getting-started/the-sx-prop/#passing-the-sx-prop
 * for more details
 *
 * @param sxs: Mui Sx props
 * @returns combined sx
 */
export const combineSx = (...sxs: SxProps<Theme>[]): SxProps<Theme> => {
  // eslint-disable-next-line @typescript-eslint/no-unsafe-return
  return sxs.map((sx) => (Array.isArray(sx) ? sx : [sx])).flat() as SxProps<Theme>;
};
