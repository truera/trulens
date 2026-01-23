import type { Meta, StoryObj } from '@storybook/react';
import RecordTree from '@/RecordTree/RecordTree';
import {
  mockDeepNode,
  mockLongDurationNode,
  mockMultipleChildrenNode,
  mockNestedNode,
  mockNodeWithOrphanedChildren,
  mockSimpleNode,
} from '@/__testing__/nodes';

type Story = StoryObj<typeof RecordTree>;

const meta: Meta<typeof RecordTree> = {
  title: 'Components/RecordTree',
  component: RecordTree,
  args: {
    root: mockSimpleNode,
    selectedNodeId: null,
    setSelectedNodeId: () => {},
  },
};

export default meta;

export const Default: Story = {};

export const WithNestedNodes: Story = {
  args: {
    root: mockNestedNode,
  },
};

export const WithSelectedNode: Story = {
  args: {
    root: mockNestedNode,
    selectedNodeId: 'node-2',
  },
};

export const WithLongDuration: Story = {
  args: {
    root: mockLongDurationNode,
  },
};

export const WithMultipleChildren: Story = {
  args: {
    root: mockMultipleChildrenNode,
  },
};

export const DeepNestedTree: Story = {
  args: {
    root: mockDeepNode,
  },
};

export const OrphanedChildren: Story = {
  args: {
    root: mockNodeWithOrphanedChildren,
  },
};
