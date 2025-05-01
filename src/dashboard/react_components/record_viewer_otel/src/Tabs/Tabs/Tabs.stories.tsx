import type { Meta, StoryObj } from '@storybook/react';
import { Box, Tab, Typography } from '@mui/material';
import Tabs from './Tabs';

const meta = {
  title: 'Components/Tabs',
  component: Tabs,
  parameters: {
    layout: 'centered',
  },
  argTypes: {
    value: {
      control: 'text',
      description: 'The value of the currently selected tab',
    },
    onChange: { action: 'onChange' },
  },
  decorators: [
    (Story) => (
      <Box sx={{ p: 3, width: '100%', maxWidth: 600 }}>
        <Story />
      </Box>
    ),
  ],
} satisfies Meta<typeof Tabs>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    value: 0,
    children: [
      <Tab key="tab1" label="Tab 1" value={0} />,
      <Tab key="tab2" label="Tab 2" value={1} />,
      <Tab key="tab3" label="Tab 3" value={2} />,
    ],
  },
  render: (args) => (
    <Box>
      <Tabs {...args} />
      <Box sx={{ p: 2, bgcolor: 'grey.100', borderRadius: 1, mt: 2 }}>
        <Typography>Tab content would appear here based on selected tab: {args.value}</Typography>
      </Box>
    </Box>
  ),
};

export const ManyTabs: Story = {
  args: {
    value: 0,
    children: Array.from({ length: 10 }, (_, i) => <Tab key={`tab${i + 1}`} label={`Tab ${i + 1}`} value={i} />),
  },
};

export const CustomStyling: Story = {
  args: {
    value: 1,
    sx: {
      bgcolor: 'background.paper',
      boxShadow: 1,
      borderRadius: 1,
      px: 2,
    },
    children: [
      <Tab key="tab1" label="Custom Tab 1" value={0} />,
      <Tab key="tab2" label="Custom Tab 2" value={1} />,
      <Tab key="tab3" label="Custom Tab 3" value={2} />,
    ],
  },
};
