import type { Meta, StoryObj } from '@storybook/react';
import RecordTable from './RecordTable';
import {
  mockSimpleNode,
  mockLongDurationNode,
  mockNestedNode,
  mockMultipleChildrenNode,
  mockNodeWithOrphanedChildren,
} from '@/__testing__/nodes';

type Story = StoryObj<typeof RecordTable>;

const meta: Meta<typeof RecordTable> = {
  title: 'Components/RecordTable',
  component: RecordTable,
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

export const WithOrphanedNodes: Story = {
  args: {
    root: mockNodeWithOrphanedChildren,
  },
};
