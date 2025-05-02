import type { Meta, StoryObj } from '@storybook/react';
import { Box, Typography } from '@mui/material';
import SpanTooltip from './SpanTooltip';
import { StackTreeNode } from '@/types/StackTreeNode';
import { createStackTreeNode } from '@/__testing__/createStackTreeNode';

const meta = {
  title: 'Components/SpanTooltip',
  component: SpanTooltip,
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
} satisfies Meta<typeof SpanTooltip>;

export default meta;
type Story = StoryObj<typeof meta>;

// Sample node data
const sampleNode: StackTreeNode = createStackTreeNode({
  id: '1',
  name: 'HTTP GET /api/users',
  startTime: 1640995200, // 2022-01-01 00:00:00
  endTime: 1640995201, // 2022-01-01 00:00:01
});

const llmNode: StackTreeNode = createStackTreeNode({
  id: '2',
  name: 'LLM Generation',
  startTime: 1640995200,
  endTime: 1640995205,
});

const unknownNode: StackTreeNode = createStackTreeNode({
  id: '3',
  name: 'Process Data',
  startTime: 1640995200,
  endTime: 1640995203,
});

export const Default: Story = {
  args: {
    node: sampleNode,
    placement: 'top',
    children: <Typography>Hover over me to see the tooltip</Typography>,
  },
};

export const LLMSpan: Story = {
  args: {
    node: llmNode,
    placement: 'right',
    children: <Typography>LLM Span (right placement)</Typography>,
  },
};

export const UnknownSpanType: Story = {
  args: {
    node: unknownNode,
    placement: 'bottom',
    children: <Typography>Unknown Span Type (bottom placement)</Typography>,
  },
};

export const CustomChild: Story = {
  args: {
    node: sampleNode,
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
