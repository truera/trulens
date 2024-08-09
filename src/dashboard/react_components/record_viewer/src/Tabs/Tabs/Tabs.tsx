// MUI Tabs use any, so we're trying to preserve the typing.
/* eslint-disable @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-explicit-any */
import { SxProps, tabClasses, Tabs as MUITabs, Theme } from '@mui/material';
import React, { PropsWithChildren } from 'react';

import { combineSx } from '@/utils/styling';

interface TabProps {
  value?: any;
  onChange?: (event: React.SyntheticEvent<Element, Event>, value: any) => void;
  sx?: SxProps<Theme>;
}

export default function Tabs({ children, value = false, onChange, sx = {} }: PropsWithChildren<TabProps>) {
  return (
    <MUITabs
      value={value}
      onChange={onChange}
      indicatorColor="primary"
      variant="scrollable"
      scrollButtons="auto"
      sx={combineSx(tabsSx, sx)}
    >
      {children}
    </MUITabs>
  );
}

const tabsSx: SxProps<Theme> = ({ spacing, palette }) => ({
  minHeight: spacing(5),
  cursor: 'pointer',
  [`& .${tabClasses.root}`]: {
    minWidth: 'auto',
    textTransform: 'none',
    minHeight: spacing(5),
    py: 0,
    borderTopLeftRadius: spacing(0.5),
    borderTopRightRadius: spacing(0.5),
  },
  [`& .${tabClasses.selected}`]: {
    backgroundColor: palette.primary.lighter,
    ':hover': {
      backgroundColor: palette.primary.lighter,
    },
  },
  '& button:hover': {
    backgroundColor: palette.grey[50],
  },
});
