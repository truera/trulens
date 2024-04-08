import React, { ReactNode } from 'react';
import { alpha, Tab, Tabs, SxProps, Theme } from '@mui/material';
import { DrawerKeyEnum } from 'components/Stores/DrawerStore';
import { combineSx } from 'util/Styling';

export type TabProps = {
  sx?: SxProps<Theme>;
  label?: ReactNode; // What the user will see
  value: DrawerKeyEnum; // Used as identifier to determine with tab is active, also used as fallback for label
  onClick?: (value?: DrawerKeyEnum) => void;
};

export type TabsProps = {
  tabs: TabProps[];
  darkMode?: boolean;
  sx?: SxProps<Theme>;
  value: DrawerKeyEnum;
  onChange?: (value: DrawerKeyEnum) => void;
};

export default function DrawerTabs({ tabs, darkMode = false, sx, value, onChange }: TabsProps) {
  const allowedValues = tabs.map(({ value: tabValue }) => tabValue);
  const selectedValue = allowedValues.indexOf(value) === -1 ? false : value;

  return (
    <Tabs
      value={selectedValue}
      indicatorColor="primary"
      sx={combineSx(tabsStyle, sx)}
      TabIndicatorProps={{ sx: selectedIndicatorStyle }}
    >
      {tabs.map((tab) => {
        const { label, value: tabValue, sx: tabSx } = tab;
        return (
          <Tab
            sx={combineSx(darkMode ? darkModeTabStyle : lightModeTabStyle, tabSx)}
            label={label}
            onClick={() => onChange?.(tabValue)}
            value={tabValue}
            key={tabValue}
          />
        );
      })}
    </Tabs>
  );
}

const defaultTabStyle: SxProps<Theme> = {
  opacity: 1,
  minWidth: 40,
  maxWidth: 40,
  minHeight: 40,
  maxHeight: 40,
  margin: 0,
  borderRadius: '50%',
  '& .MuiTab-wrapper': {
    height: 48,
    width: 48,
    padding: 12,
    borderRadius: '50%',
    '&:hover': {
      backgroundColor: ({ palette }) => alpha(palette.action.active, palette.action.hoverOpacity),
      // Reset on touch devices, it doesn't add specificity
      '@media (hover: none)': {
        backgroundColor: 'transparent',
      },
    },
  },
};

const lightModeTabStyle: SxProps<Theme> = {
  ...defaultTabStyle,
  '&:hover': {
    backgroundColor: ({ palette }) => palette.grey[200],
  },
};

const darkModeTabStyle: SxProps<Theme> = {
  ...defaultTabStyle,
  '&:hover': {
    backgroundColor: ({ palette }) => alpha(palette.primary.main, palette.action.hoverOpacity),
  },
};

const tabsStyle: SxProps<Theme> = {
  height: '100%',
  overflow: 'inherit',
  '& .MuiTabs-flexContainer': {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
  },
  display: 'flex',
  alignItems: 'center',
  marginRight: '4px',
  '& .MuiTabs-fixed': {
    height: '100%',
    display: 'flex',
  },
  '& span': {
    display: 'none',
  },
};

const selectedIndicatorStyle: SxProps<Theme> = {
  height: 4,
};
