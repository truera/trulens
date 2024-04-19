import { Box, SxProps, Theme, Typography } from '@mui/material';
import { ReactNode } from 'react';

import { combineSx } from '@/utils/styling';

interface LabelAndValueProps {
  label: string;
  value: ReactNode;
  align?: 'start' | 'center' | 'end';
  sx?: SxProps<Theme>;
}

export const LabelAndValueClasses = {
  value: 'label-and-value-value',
};

export default function LabelAndValue({ label, value, align = 'start', sx = {} }: LabelAndValueProps) {
  return (
    <Box sx={combineSx(containerSx, sx)}>
      <Typography variant="subtitle1" sx={labelSx}>
        {label}
      </Typography>
      <Box
        className={LabelAndValueClasses.value}
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: align,
        }}
      >
        {value}
      </Box>
    </Box>
  );
}

const labelSx: SxProps<Theme> = {
  display: 'flex',
  flexDirection: 'column',
  fontWeight: ({ typography }) => typography.fontWeightBold,
};

const containerSx: SxProps = {
  display: 'flex',
  flexDirection: 'column',
  marginRight: 3,
};
