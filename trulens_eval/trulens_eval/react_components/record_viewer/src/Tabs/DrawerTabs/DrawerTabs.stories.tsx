import React, { useState } from 'react';
import { HelpOutline } from '@mui/icons-material';
import { Box } from '@mui/material';
import { DrawerKeyEnum } from 'components/Stores/DrawerStore';
import { DrawerTabs } from 'components/UI/Navigation/Tabs';
import DrawerTabsDocumentation from 'components/UI/Navigation/Tabs/DrawerTabs/DrawerTabs.mdx';
import { StyledTooltip } from 'components/UI/StyledTooltip';

export default {
  title: 'Components/Navigation/Tabs/Drawer Tabs',
  parameters: {
    docs: {
      page: DrawerTabsDocumentation,
    },
  },
};

export const Example = () => {
  const [value, setSelectedValue] = useState<DrawerKeyEnum>();

  const helpTab = {
    label: (
      <StyledTooltip title="Open help center">
        <HelpOutline />
      </StyledTooltip>
    ),
    value: DrawerKeyEnum.help,
  };

  const tabs = [helpTab];

  return (
    <Box width="100%" height="64px">
      <DrawerTabs tabs={tabs} value={value} onChange={setSelectedValue} />
    </Box>
  );
};
