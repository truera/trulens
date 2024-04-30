import { SxProps, tableBodyClasses, tableCellClasses, Theme } from '@mui/material';

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

export const tableWithoutBorderSx: SxProps<Theme> = {
  borderRadius: ({ spacing }) => spacing(0.5),
  minWidth: 650,

  '& th': {
    backgroundColor: ({ palette }) => palette.grey[100],
    color: ({ palette }) => palette.grey[600],
    fontWeight: 600,
  },

  [`& .${tableCellClasses.root}`]: {
    borderRight: ({ palette }) => `1px solid ${palette.grey[300]}`,
  },

  [`& .${tableCellClasses.root}:last-child`]: {
    borderRight: 'none',
  },

  [`& .${tableBodyClasses.root} .${tableCellClasses.root}`]: {
    mx: 1,
  },
};

export const tableWithBorderSx: SxProps<Theme> = {
  ...tableWithoutBorderSx,
  border: ({ palette }) => `1px solid ${palette.grey[300]}`,
};
