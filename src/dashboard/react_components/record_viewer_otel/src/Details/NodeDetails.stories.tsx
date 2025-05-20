import type { Meta, StoryObj } from '@storybook/react';
import NodeDetails from './NodeDetails';
import { SpanAttributes, SpanType } from '@/constants/span';
import { createStackTreeNode } from '@/__testing__/createStackTreeNode';
import { Stack } from '@mui/material';

type Story = StoryObj<typeof NodeDetails>;

const meta: Meta<typeof NodeDetails> = {
  title: 'Components/NodeDetails',
  component: NodeDetails,
  args: {
    selectedNode: createStackTreeNode({
      id: 'node-1',
      name: 'Test Node',
      startTime: 0,
      endTime: 0.15, // 150ms in seconds
      attributes: {
        [SpanAttributes.SPAN_TYPE]: SpanType.CUSTOM,
        [SpanAttributes.INPUT_ID]: 'input-123',
        'custom.attribute': 'value',
      },
    }),
  },
  decorators: [
    (Story) => (
      <Stack gap={2}>
        <Story />
      </Stack>
    ),
  ],
};

export default meta;

export const Default: Story = {};

export const GenerationSpanType: Story = {
  args: {
    selectedNode: createStackTreeNode({
      startTime: 0,
      endTime: 0.25, // 250ms in seconds
      attributes: {
        [SpanAttributes.SPAN_TYPE]: SpanType.GENERATION,
        [SpanAttributes.INPUT_ID]: 'input-123',
        [SpanAttributes.GENERATION_MODEL_NAME]: 'gpt-4',
        [SpanAttributes.GENERATION_INPUT_MESSAGES]: 'Tell me about AI',
        [SpanAttributes.GENERATION_OUTPUT_MESSAGES]: 'AI is an exciting field...',
      },
    }),
  },
};

export const ToolInvocationSpanType: Story = {
  args: {
    selectedNode: createStackTreeNode({
      startTime: 0,
      endTime: 0.12, // 120ms in seconds
      attributes: {
        [SpanAttributes.SPAN_TYPE]: SpanType.TOOL_INVOCATION,
        [SpanAttributes.INPUT_ID]: 'input-456',
        [SpanAttributes.TOOL_INVOCATION_DESCRIPTION]: 'web_search',
        [SpanAttributes.CALL_ARGS]: '{"query": "latest AI developments"}',
        [SpanAttributes.CALL_RETURN]: '{"results": ["Result 1", "Result 2"]}',
      },
    }),
  },
};

export const WithoutSpanType: Story = {
  args: {
    selectedNode: createStackTreeNode({
      startTime: 0,
      endTime: 0.08, // 80ms in seconds
      attributes: {
        [SpanAttributes.INPUT_ID]: 'input-789',
        'custom.attribute1': 'value1',
        'custom.attribute2': 'value2',
      },
    }),
  },
};

export const WithoutInputId: Story = {
  args: {
    selectedNode: createStackTreeNode({
      startTime: 0,
      endTime: 0.18, // 180ms in seconds
      attributes: {
        [SpanAttributes.SPAN_TYPE]: SpanType.CUSTOM,
        [SpanAttributes.SELECTOR_NAME]: 'Processing Chain',
      },
    }),
  },
};

export const ManyAttributes: Story = {
  args: {
    selectedNode: createStackTreeNode({
      startTime: 0,
      endTime: 0.35, // 350ms in seconds
      attributes: {
        [SpanAttributes.SPAN_TYPE]: SpanType.GENERATION,
        [SpanAttributes.INPUT_ID]: 'input-abc',
        [SpanAttributes.GENERATION_MODEL_NAME]: 'gpt-3.5-turbo',
        [SpanAttributes.GENERATION_TEMPERATURE]: '0.7',
        [SpanAttributes.GENERATION_INPUT_TOKEN_COUNT]: '1024',
        [SpanAttributes.GENERATION_INPUT_MESSAGES]: 'A very long prompt that would likely span multiple lines...',
        [SpanAttributes.GENERATION_OUTPUT_MESSAGES]: 'An equally long completion response that has many details...',
        [SpanAttributes.APP_VERSION]: 'value1',
        [SpanAttributes.DOMAIN]: 'value2',
        [SpanAttributes.APP_NAME]: 'value3',
      },
    }),
  },
};
