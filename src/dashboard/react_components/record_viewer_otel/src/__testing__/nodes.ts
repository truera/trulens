import { StackTreeNode } from '@/types/StackTreeNode';
import { createStackTreeNode } from './createStackTreeNode';
import { SpanAttributes, SpanType } from '@/constants/span';

let currentNode = createStackTreeNode({
  id: 'deep-1',
  name: 'Root Operation',
  startTime: 0,
  endTime: 2,
  attributes: { [SpanAttributes.SPAN_TYPE]: SpanType.RECORD_ROOT },
  children: [],
});

const mockDeepNode = currentNode;
const nodeMap: Record<string, StackTreeNode> = {};
nodeMap[mockDeepNode.id] = mockDeepNode;

for (let i = 2; i <= 25; i++) {
  const childNode = createStackTreeNode({
    id: `deep-${i}`,
    name: `Layer ${i} Operation`,
    startTime: (i - 1) * 0.03,
    endTime: 2 - (i - 1) * 0.03,
    attributes: {
      [SpanAttributes.SPAN_TYPE]:
        i % 5 === 0
          ? SpanType.RETRIEVAL
          : i % 4 === 0
          ? SpanType.GENERATION
          : i % 3 === 0
          ? SpanType.RERANKING
          : i % 2 === 0
          ? SpanType.TOOL_INVOCATION
          : SpanType.AGENT_INVOCATION,
    },
    children: [],
    parentId: currentNode.id,
  });

  nodeMap[childNode.id] = childNode;
  currentNode.children.push(childNode);
  currentNode = childNode;
}

export const mockSimpleNode = createStackTreeNode({
  id: 'node-1',
  name: 'GET /api/data',
  startTime: 0,
  endTime: 0.15,
  attributes: {
    [SpanAttributes.SPAN_TYPE]: SpanType.RECORD_ROOT,
    [SpanAttributes.CALL_FUNCTION]: 'GET',
    [SpanAttributes.CALL_SIGNATURE]: '/api/data',
    [SpanAttributes.CALL_RETURN]: 200,
  },
  children: [],
});

export const mockNestedNode = createStackTreeNode({
  id: 'node-1',
  name: 'GET /api/users',
  startTime: 0,
  endTime: 0.35,
  attributes: {
    [SpanAttributes.SPAN_TYPE]: SpanType.RECORD_ROOT,
    [SpanAttributes.CALL_FUNCTION]: 'GET',
    [SpanAttributes.CALL_SIGNATURE]: '/api/users',
    [SpanAttributes.CALL_RETURN]: 200,
  },
  children: [
    createStackTreeNode({
      id: 'node-2',
      name: 'Database Query',
      startTime: 0.05,
      endTime: 0.25,
      attributes: {
        [SpanAttributes.SPAN_TYPE]: SpanType.RETRIEVAL,
        [SpanAttributes.RETRIEVAL_QUERY_TEXT]: 'SELECT * FROM users',
      },
      children: [
        createStackTreeNode({
          id: 'node-3',
          name: 'Select Users',
          startTime: 0.08,
          endTime: 0.2,
          attributes: {
            [SpanAttributes.SPAN_TYPE]: SpanType.RETRIEVAL,
            [SpanAttributes.SELECTOR_NAME]: 'user_retrieval',
          },
          children: [],
        }),
      ],
      parentId: 'node-1',
    }),
  ],
});

export const mockComplexNode = createStackTreeNode({
  id: 'node-1',
  name: 'API Request',
  startTime: 0,
  endTime: 0.6,
  attributes: {
    [SpanAttributes.SPAN_TYPE]: SpanType.RECORD_ROOT,
    [SpanAttributes.CALL_FUNCTION]: 'POST',
    [SpanAttributes.CALL_SIGNATURE]: '/api/process',
    [SpanAttributes.CALL_RETURN]: 201,
  },
  children: [
    createStackTreeNode({
      id: 'node-2',
      name: 'Authentication',
      startTime: 0.02,
      endTime: 0.12,
      attributes: {
        [SpanAttributes.SPAN_TYPE]: SpanType.CUSTOM,
        [SpanAttributes.CALL_CLASS]: 'JWT',
      },
      children: [],
      parentId: 'node-1',
    }),
    createStackTreeNode({
      id: 'node-3',
      name: 'Database Query',
      startTime: 0.13,
      endTime: 0.38,
      attributes: {
        [SpanAttributes.SPAN_TYPE]: SpanType.RETRIEVAL,
        [SpanAttributes.RETRIEVAL_QUERY_TEXT]: 'insert',
      },
      children: [],
      parentId: 'node-1',
    }),
    createStackTreeNode({
      id: 'node-4',
      name: 'Response Processing',
      startTime: 0.39,
      endTime: 0.54,
      attributes: {
        [SpanAttributes.SPAN_TYPE]: SpanType.GENERATION,
        [SpanAttributes.GENERATION_MODEL_NAME]: 'JsonFormatter',
      },
      children: [],
      parentId: 'node-1',
    }),
  ],
});

export const mockMultipleChildrenNode = createStackTreeNode({
  id: 'node-1',
  name: 'API Request',
  startTime: 0,
  endTime: 0.6,
  attributes: { [SpanAttributes.SPAN_TYPE]: SpanType.RECORD_ROOT },
  children: [
    createStackTreeNode({
      id: 'node-2',
      name: 'Authentication',
      startTime: 0.02,
      endTime: 0.12,
      attributes: { [SpanAttributes.SPAN_TYPE]: SpanType.CUSTOM },
      children: [],
      parentId: 'node-1',
    }),
    createStackTreeNode({
      id: 'node-3',
      name: 'Database Query',
      startTime: 0.13,
      endTime: 0.38,
      attributes: { [SpanAttributes.SPAN_TYPE]: SpanType.RETRIEVAL },
      children: [],
      parentId: 'node-1',
    }),
    createStackTreeNode({
      id: 'node-4',
      name: 'Response Processing',
      startTime: 0.39,
      endTime: 0.54,
      attributes: { [SpanAttributes.SPAN_TYPE]: SpanType.GENERATION },
      children: [],
      parentId: 'node-1',
    }),
  ],
});

export const mockLongDurationNode = createStackTreeNode({
  id: 'node-1',
  name: 'Process Data',
  startTime: 0,
  endTime: 1.5,
  attributes: { [SpanAttributes.SPAN_TYPE]: SpanType.RECORD_ROOT },
  children: [
    createStackTreeNode({
      id: 'node-2',
      name: 'Heavy Computation',
      startTime: 0.1,
      endTime: 1.4,
      attributes: { [SpanAttributes.SPAN_TYPE]: SpanType.GENERATION },
      children: [],
      parentId: 'node-1',
    }),
  ],
});

export { mockDeepNode };
