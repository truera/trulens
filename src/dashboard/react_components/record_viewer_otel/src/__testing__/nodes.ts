import { StackTreeNode } from '@/types/StackTreeNode';
import { createStackTreeNode } from './createStackTreeNode';
import { SpanAttributes } from '@/constants/span';

// Create a deep tree with over 20 layers
let currentNode = createStackTreeNode({
  id: 'deep-1',
  name: 'Root Operation',
  startTime: 0,
  endTime: 2000,
  attributes: { [SpanAttributes.SPAN_TYPE]: 'application' },
  children: [],
});

const deepNode = currentNode;
const nodeMap: Record<string, StackTreeNode> = {};
nodeMap[deepNode.id] = deepNode;

// Create 25 layers of nesting
for (let i = 2; i <= 25; i++) {
  const childNode = createStackTreeNode({
    id: `deep-${i}`,
    name: `Layer ${i} Operation`,
    startTime: (i - 1) * 30,
    endTime: 2000 - (i - 1) * 30,
    attributes: {
      [SpanAttributes.SPAN_TYPE]:
        i % 5 === 0
          ? 'database'
          : i % 4 === 0
          ? 'http'
          : i % 3 === 0
          ? 'processing'
          : i % 2 === 0
          ? 'computation'
          : 'function',
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
  endTime: 150,
  attributes: {
    [SpanAttributes.SPAN_TYPE]: 'http',
    'http.method': 'GET',
    'http.path': '/api/data',
    'http.status_code': 200,
  },
  children: [],
});

export const mockNestedNode = createStackTreeNode({
  id: 'node-1',
  name: 'GET /api/users',
  startTime: 0,
  endTime: 350,
  attributes: {
    [SpanAttributes.SPAN_TYPE]: 'http',
    'http.method': 'GET',
    'http.path': '/api/users',
    'http.status_code': 200,
  },
  children: [
    createStackTreeNode({
      id: 'node-2',
      name: 'Database Query',
      startTime: 50,
      endTime: 250,
      attributes: {
        [SpanAttributes.SPAN_TYPE]: 'database',
        'db.system': 'postgresql',
        'db.statement': 'SELECT * FROM users',
      },
      children: [
        createStackTreeNode({
          id: 'node-3',
          name: 'Select Users',
          startTime: 80,
          endTime: 200,
          attributes: {
            [SpanAttributes.SPAN_TYPE]: 'query',
            'query.name': 'user_retrieval',
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
  endTime: 600,
  attributes: {
    [SpanAttributes.SPAN_TYPE]: 'http',
    'http.method': 'POST',
    'http.path': '/api/process',
    'http.status_code': 201,
  },
  children: [
    createStackTreeNode({
      id: 'node-2',
      name: 'Authentication',
      startTime: 20,
      endTime: 120,
      attributes: {
        [SpanAttributes.SPAN_TYPE]: 'auth',
        'auth.method': 'JWT',
      },
      children: [],
      parentId: 'node-1',
    }),
    createStackTreeNode({
      id: 'node-3',
      name: 'Database Query',
      startTime: 130,
      endTime: 380,
      attributes: {
        [SpanAttributes.SPAN_TYPE]: 'database',
        'db.system': 'mongodb',
        'db.operation': 'insert',
      },
      children: [],
      parentId: 'node-1',
    }),
    createStackTreeNode({
      id: 'node-4',
      name: 'Response Processing',
      startTime: 390,
      endTime: 540,
      attributes: {
        [SpanAttributes.SPAN_TYPE]: 'processing',
        'processor.name': 'JsonFormatter',
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

export const mockLongDurationNode = createStackTreeNode({
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

export { deepNode };
