import { SxProps, Theme } from '@mui/material';

export const summarySx: SxProps<Theme> = {
  border: ({ vars }) => `1px solid ${vars.palette.grey[300]}`,
  pl: 2,
  py: 1,
  borderRadius: 0.5,
  width: 'fit-content',
};
