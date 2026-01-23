import type { Meta, StoryObj } from '@storybook/react';
import { Box, Typography, Button } from '@mui/material';
import StyledTooltip from './StyledTooltip';

const meta = {
  title: 'Components/StyledTooltip',
  component: StyledTooltip,
  parameters: {
    layout: 'centered',
  },
  argTypes: {
    placement: {
      control: 'select',
      options: ['top', 'bottom', 'left', 'right', 'top-start', 'top-end', 'bottom-start', 'bottom-end'],
      description: 'The placement of the tooltip',
      defaultValue: 'top',
    },
  },
  decorators: [
    (Story) => (
      <Box sx={{ p: 3 }}>
        <Story />
      </Box>
    ),
  ],
} satisfies Meta<typeof StyledTooltip>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    title: 'This is a styled tooltip',
    placement: 'top',
    children: <Typography>Hover over me to see the tooltip</Typography>,
  },
};

export const WithButton: Story = {
  args: {
    title: 'Click to perform an action',
    placement: 'right',
    children: <Button variant="contained">Button with tooltip</Button>,
  },
};

export const LongContent: Story = {
  args: {
    title: 'This is a tooltip with a longer description that might wrap to multiple lines for demonstration purposes.',
    placement: 'bottom',
    children: <Typography>Hover for longer tooltip</Typography>,
  },
};

export const WithRichContent: Story = {
  args: {
    title: (
      <Box sx={{ p: 1 }}>
        <Typography variant="subtitle2">Rich Tooltip Content</Typography>
        <Typography variant="body2">This tooltip contains structured content with different elements.</Typography>
        <Box sx={{ mt: 1, fontSize: '0.75rem', color: 'text.secondary' }}>Additional information here</Box>
      </Box>
    ),
    placement: 'left',
    children: (
      <Box
        sx={{
          width: 150,
          height: 50,
          bgcolor: 'primary.main',
          color: 'white',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          borderRadius: 1,
          cursor: 'pointer',
        }}
      >
        Custom Element
      </Box>
    ),
  },
};
