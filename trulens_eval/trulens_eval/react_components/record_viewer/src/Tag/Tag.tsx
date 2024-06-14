import { AlertColor, Box, SxProps, Theme, Typography } from '@mui/material';
import { PropsWithChildren, ReactNode } from 'react';

import { combineSx } from '@/utils/styling';

export interface TagProps {
  severity: AlertColor;
  title: string;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  sx?: SxProps<Theme>;
}

export default function Tag({ severity, title, leftIcon, rightIcon, sx = {} }: PropsWithChildren<TagProps>) {
  return (
    <Box sx={combineSx(tagSx(severity), sx)}>
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
  borderRadius: ({ spacing }) => spacing(0.5),
  width: 'fit-content',
  height: 'fit-content',
};
const tagSx = (severity: AlertColor): SxProps<Theme> => {
  if (severity === 'info') {
    return {
      ...tagMainContainerStyles,
      border: ({ palette }) => `1px solid ${palette.grey[300]}`,
      background: ({ palette }) => palette.grey[100],
    };
  }

  return {
    ...tagMainContainerStyles,
    border: ({ palette }) => `1px solid ${palette[severity].main}`,
    background: ({ palette }) => palette[severity].light,
  };
};

const titleStyle: SxProps<Theme> = {
  color: ({ palette }) => palette.grey[900],
  fontWeight: ({ typography }) => typography.fontWeightBold,
  alignSelf: 'center',
  overflow: 'auto',
};

const leftIconStyle: SxProps<Theme> = {
  paddingRight: '4px',
  display: 'flex',
};
