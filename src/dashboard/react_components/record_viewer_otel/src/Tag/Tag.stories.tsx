import type { Meta, StoryObj } from '@storybook/react';
import { Box } from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import CancelIcon from '@mui/icons-material/Cancel';
import Tag from './Tag';

const meta = {
  title: 'Components/Tag',
  component: Tag,
  parameters: {
    layout: 'centered',
  },
  argTypes: {
    title: {
      control: 'text',
      description: 'The text content of the tag',
    },
    leftIcon: {
      control: 'boolean',
      description: 'Optional icon displayed on the left side of the tag',
    },
    rightIcon: {
      control: 'boolean',
      description: 'Optional icon displayed on the right side of the tag',
    },
  },
  decorators: [
    (Story) => (
      <Box sx={{ p: 3, width: '100%', maxWidth: 600 }}>
        <Story />
      </Box>
    ),
  ],
} satisfies Meta<typeof Tag>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    title: 'Simple Tag',
  },
};

export const WithLeftIcon: Story = {
  args: {
    title: 'Info Tag',
    leftIcon: <InfoIcon fontSize="small" />,
  },
};

export const WithRightIcon: Story = {
  args: {
    title: 'Dismissible Tag',
    rightIcon: <CancelIcon fontSize="small" />,
  },
};

export const WithBothIcons: Story = {
  args: {
    title: 'Complete Tag',
    leftIcon: <InfoIcon fontSize="small" />,
    rightIcon: <CancelIcon fontSize="small" />,
  },
};

export const CustomStyling: Story = {
  args: {
    title: 'Custom Tag',
    leftIcon: <InfoIcon fontSize="small" color="primary" />,
    sx: {
      bgcolor: 'primary.light',
      color: 'primary.contrastText',
      border: '1px solid',
      borderColor: 'primary.main',
    },
  },
};

export const LongText: Story = {
  args: {
    title: 'This is a tag with very long text content that might need to wrap or be truncated',
    sx: {
      maxWidth: '200px',
    },
  },
};
