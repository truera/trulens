import type { Meta, StoryObj } from '@storybook/react';
import TracePanel from './TracePanel';
import { createStackTreeNode } from '@/__testing__/createStackTreeNode';
import { SpanAttributes } from '@/constants/span';

type Story = StoryObj<typeof TracePanel>;

// Mock data
const mockBasicNode = createStackTreeNode({
  id: 'trace-1',
  name: 'Basic Record',
  startTime: 0,
  endTime: 150,
  attributes: {
    [SpanAttributes.SPAN_TYPE]: 'record',
    [SpanAttributes.RECORD_ROOT_INPUT]: 'What is the capital of France?',
    [SpanAttributes.RECORD_ROOT_OUTPUT]: 'The capital of France is Paris.',
  },
  children: [],
});

const mockComplexNode = createStackTreeNode({
  id: 'trace-2',
  name: 'Complex Record',
  startTime: 0,
  endTime: 350,
  attributes: {
    [SpanAttributes.SPAN_TYPE]: 'record',
    [SpanAttributes.RECORD_ROOT_INPUT]: 'Explain the process of photosynthesis in detail.',
    [SpanAttributes.RECORD_ROOT_OUTPUT]:
      'Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy. The process happens in the chloroplasts of plant cells, primarily using chlorophyll to capture light energy. This energy is used to convert carbon dioxide and water into glucose and oxygen. The overall equation is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2.',
  },
  children: [],
});

const mockErrorNode = createStackTreeNode({
  id: 'trace-3',
  name: 'Error Record',
  startTime: 0,
  endTime: 250,
  attributes: {
    [SpanAttributes.SPAN_TYPE]: 'record',
    [SpanAttributes.RECORD_ROOT_INPUT]: 'Generate a comprehensive market analysis',
    [SpanAttributes.RECORD_ROOT_ERROR]: 'Error: API rate limit exceeded. Please try again later.',
  },
  children: [],
});

const mockEmptyNode = createStackTreeNode({
  id: 'trace-4',
  name: 'Empty Record',
  startTime: 0,
  endTime: 100,
  attributes: {
    [SpanAttributes.SPAN_TYPE]: 'record',
  },
  children: [],
});

const meta: Meta<typeof TracePanel> = {
  title: 'Components/TracePanel',
  component: TracePanel,
  args: {
    root: mockBasicNode,
  },
};

export default meta;

export const Default: Story = {};

export const WithComplexData: Story = {
  args: {
    root: mockComplexNode,
  },
};

export const WithError: Story = {
  args: {
    root: mockErrorNode,
  },
};

export const WithEmptyData: Story = {
  args: {
    root: mockEmptyNode,
  },
};
