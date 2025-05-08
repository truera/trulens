import type { Meta, StoryObj } from '@storybook/react';
import RecordTree from './RecordTree';
import { createStackTreeNode } from '@/__testing__/createStackTreeNode';
import { SpanAttributes } from '@/constants/span';
import { StackTreeNode } from '@/types/StackTreeNode';

type Story = StoryObj<typeof RecordTree>;

// Mock data
const mockSimpleNode = createStackTreeNode({
  id: 'node-1',
  name: 'GET /api/data',
  startTime: 0,
  endTime: 150,
  attributes: { [SpanAttributes.SPAN_TYPE]: 'http' },
  children: [],
});

const mockNestedNode = createStackTreeNode({
  id: 'node-1',
  name: 'GET /api/users',
  startTime: 0,
  endTime: 350,
  attributes: { [SpanAttributes.SPAN_TYPE]: 'http' },
  children: [
    createStackTreeNode({
      id: 'node-2',
      name: 'Database Query',
      startTime: 50,
      endTime: 250,
      attributes: { [SpanAttributes.SPAN_TYPE]: 'database' },
      children: [
        createStackTreeNode({
          id: 'node-3',
          name: 'Select Users',
          startTime: 80,
          endTime: 200,
          attributes: { [SpanAttributes.SPAN_TYPE]: 'query' },
          children: [],
        }),
      ],
      parentId: 'node-1',
    }),
  ],
});

const mockLongDurationNode = createStackTreeNode({
  id: 'node-1',
  name: 'Process Data',
  startTime: 0,
  endTime: 1500,
  attributes: { [SpanAttributes.SPAN_TYPE]: 'function' },
  children: [
    createStackTreeNode({
      id: 'node-2',
      name: 'Heavy Computation',
      startTime: 100,
      endTime: 1400,
      attributes: { [SpanAttributes.SPAN_TYPE]: 'computation' },
      children: [],
      parentId: 'node-1',
    }),
  ],
});

const mockMultipleChildrenNode = createStackTreeNode({
  id: 'node-1',
  name: 'API Request',
  startTime: 0,
  endTime: 600,
  attributes: { [SpanAttributes.SPAN_TYPE]: 'http' },
  children: [
    createStackTreeNode({
      id: 'node-2',
      name: 'Authentication',
      startTime: 20,
      endTime: 120,
      attributes: { [SpanAttributes.SPAN_TYPE]: 'auth' },
      children: [],
      parentId: 'node-1',
    }),
    createStackTreeNode({
      id: 'node-3',
      name: 'Database Query',
      startTime: 130,
      endTime: 380,
      attributes: { [SpanAttributes.SPAN_TYPE]: 'database' },
      children: [],
      parentId: 'node-1',
    }),
    createStackTreeNode({
      id: 'node-4',
      name: 'Response Processing',
      startTime: 390,
      endTime: 540,
      attributes: { [SpanAttributes.SPAN_TYPE]: 'processing' },
      children: [],
      parentId: 'node-1',
    }),
  ],
});

// Create nodeMap for each scenario
function createNodeMap(root: StackTreeNode): Record<string, StackTreeNode> {
  const nodeMap: Record<string, StackTreeNode> = {};

  function addToMap(node: StackTreeNode) {
    nodeMap[node.id] = node;
    node.children.forEach((child) => addToMap(child));
  }

  addToMap(root);
  return nodeMap;
}

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
