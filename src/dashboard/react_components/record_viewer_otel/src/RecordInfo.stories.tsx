import type { Meta, StoryObj } from '@storybook/react';
import RecordInfo from '@/RecordInfo';
import { createNodeMap } from '@/functions/createNodeMap';
import { deepNode, mockNestedNode, mockSimpleNode, mockComplexNode } from '@/__testing__/nodes';

type Story = StoryObj<typeof RecordInfo>;

const meta: Meta<typeof RecordInfo> = {
  title: 'Components/RecordInfo',
  component: RecordInfo,
  args: {
    root: mockSimpleNode,
    nodeMap: createNodeMap(mockSimpleNode),
  },
  parameters: {
    layout: 'fullscreen',
  },
};

export default meta;

export const Default: Story = {};

export const WithNestedNodes: Story = {
  args: {
    root: mockNestedNode,
    nodeMap: createNodeMap(mockNestedNode),
  },
};

export const ComplexTree: Story = {
  args: {
    root: mockComplexNode,
    nodeMap: createNodeMap(mockComplexNode),
  },
};

export const DeepNestedTree: Story = {
  args: {
    root: deepNode,
    nodeMap: createNodeMap(deepNode),
  },
};
