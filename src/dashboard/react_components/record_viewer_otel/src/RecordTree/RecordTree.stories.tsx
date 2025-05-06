import type { Meta, StoryObj } from '@storybook/react';
import RecordTree from '@/RecordTree/RecordTree';
import { createNodeMap } from '@/functions/createNodeMap';
import {
  deepNode,
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
    nodeMap: createNodeMap(mockSimpleNode),
    selectedNodeId: null,
    setSelectedNodeId: () => {},
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

export const WithSelectedNode: Story = {
  args: {
    root: mockNestedNode,
    nodeMap: createNodeMap(mockNestedNode),
    selectedNodeId: 'node-2',
  },
};

export const WithLongDuration: Story = {
  args: {
    root: mockLongDurationNode,
    nodeMap: createNodeMap(mockLongDurationNode),
  },
};

export const WithMultipleChildren: Story = {
  args: {
    root: mockMultipleChildrenNode,
    nodeMap: createNodeMap(mockMultipleChildrenNode),
  },
};

export const DeepNestedTree: Story = {
  args: {
    root: deepNode,
    nodeMap: createNodeMap(deepNode),
  },
};

export const OrphanedChildren: Story = {
  args: {
    root: mockNodeWithOrphanedChildren,
    nodeMap: createNodeMap(mockNodeWithOrphanedChildren),
  },
};
