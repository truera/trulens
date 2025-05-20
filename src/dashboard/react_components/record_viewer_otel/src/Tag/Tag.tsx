import { Box, SxProps, Theme, Typography } from '@mui/material';
import { PropsWithChildren, ReactNode } from 'react';

import { combineSx } from '@/utils/styling';

export interface TagProps {
  title: string;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  sx?: SxProps<Theme>;
}

export default function Tag({ title, leftIcon, rightIcon, sx = {} }: PropsWithChildren<TagProps>) {
  return (
    <Box sx={combineSx(tagSx(), sx)}>
      {leftIcon && <Box sx={leftIconStyle}>{leftIcon}</Box>}
      <Typography variant="subtitle1" sx={titleStyle}>
        {title}
      </Typography>
      {rightIcon && <Box sx={{ paddingLeft: '4px', display: 'flex' }}>{rightIcon}</Box>}
    </Box>
  );
}

const tagMainContainerStyles: SxProps<Theme> = {
  display: 'flex',
  flexDirection: 'row',
  padding: (theme) => theme.spacing(1 / 2, 1),
  borderRadius: '4px',
  width: 'fit-content',
  height: 'fit-content',
};
const tagSx = (): SxProps<Theme> => {
  return {
    ...tagMainContainerStyles,
    border: ({ vars }) => `1px solid ${vars.palette.grey[300]}`,
    background: ({ vars }) => vars.palette.grey[100],
  };
};

const titleStyle: SxProps<Theme> = {
  color: ({ vars }) => vars.palette.grey[900],
  fontWeight: ({ typography }) => typography.fontWeightBold,
  alignSelf: 'center',
  overflow: 'auto',
};

const leftIconStyle: SxProps<Theme> = {
  paddingRight: '4px',
  display: 'flex',
};
