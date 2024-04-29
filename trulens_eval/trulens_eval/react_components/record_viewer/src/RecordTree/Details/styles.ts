import { SxProps, Theme } from '@mui/material';

export const summarySx: SxProps<Theme> = {
  border: ({ palette }) => `1px solid ${palette.grey[300]}`,
  pl: 2,
  py: 1,
  borderRadius: ({ spacing }) => spacing(0.5),
  width: 'fit-content',
};
