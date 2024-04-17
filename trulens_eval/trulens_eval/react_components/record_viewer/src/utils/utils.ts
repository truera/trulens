import { v4 } from 'uuid';
import { CallJSONRaw, PerfJSONRaw, RecordJSONRaw, StackJSONRaw, StackTreeNode } from './types';

export const ROOT_NODE_ID = 'root-root-root';

/**
 * Gets the name of the calling class in the stack cell.
 *
 * @param stackCell - StackJSONRaw Cell in the stack of a call.
 * @returns name of the calling class in the stack cell.
 */
export const getClassNameFromCell = (stackCell: StackJSONRaw) => {
  return stackCell.method.obj.cls.name;
};

/**
 * Gets the name of the calling method in the stack cell.
 *
 * @param stackCell - StackJSONRaw Cell in the stack of a call.
 * @returns name of the calling method in the stack cell.
 */
export const getMethodNameFromCell = (stackCell: StackJSONRaw) => {
  return stackCell.method.name;
};

/**
 * Gets the path of the calling method in the stack cell.
 *
 * @param stackCell - StackJSONRaw Cell in the stack of a call.
 * @returns name of the calling method in the stack cell.
 */
export const getPathName = (stackCell: StackJSONRaw) => {
  if (typeof stackCell.path === 'string') {
    return stackCell.path;
  }
  return stackCell.path.path
    .map((p) => {
      if (!p) return undefined;

      if ('item_or_attribute' in p) {
        return `.${p.item_or_attribute}`;
      }

      if ('index' in p) {
        return `[${p.index}]`;
      }

      return undefined;
    })
    .filter(Boolean)
    .join('');
};

/**
 * Gets the start and end times based on the performance
 * data provided.
 *
 * @param perf - PerfJSONRaw The performance data to be analyzed
 * @returns an object containing the start and end times based on the performance
 * data provided as numbers or undefined
 */
export const getStartAndEndTimes = (perf: PerfJSONRaw) => {
  return {
    startTime: perf?.start_time ? new Date(perf.start_time).getTime() : 0,
    endTime: perf?.end_time ? new Date(perf.end_time).getTime() : 0,
  };
};

const addCallToTree = (tree: StackTreeNode, call: CallJSONRaw, stack: StackJSONRaw[], index: number) => {
  const stackCell = stack[index];

  // eslint-disable-next-line no-param-reassign
  if (!tree.children) tree.children = [];

  // otherwise, we are deciding which node to go in
  let matchingNode = tree.children.find(
    (node) =>
      node.name === getClassNameFromCell(stackCell) &&
      node.startTime <= new Date(call.perf.start_time).getTime() &&
      (node.endTime === 0 || node.endTime >= new Date(call.perf.end_time).getTime())
  );

  const path = getPathName(stackCell);
  const name = getClassNameFromCell(stackCell);
  const methodName = getMethodNameFromCell(stackCell);
  const { id } = stackCell.method.obj;
  const nodeId = `${id}-${methodName}-${name}`;

  // if we are currently at the top most cell of the stack
  if (index === stack.length - 1) {
    const { startTime, endTime } = getStartAndEndTimes(call.perf);

    if (matchingNode) {
      const matchingNodeId = call.stack[index].method.obj.id;

      matchingNode.startTime = startTime;
      matchingNode.endTime = endTime;
      matchingNode.timeTaken = endTime - startTime;
      matchingNode.id = matchingNodeId;
      matchingNode.nodeId = `${matchingNodeId}-${methodName}-${name}-${startTime ?? ''}-${endTime ?? ''}`;
      matchingNode.raw = call;

      return;
    }

    tree.children.push({
      children: [],
      name,
      path,
      methodName,
      nodeId,
      id,
      startTime,
      endTime,
      timeTaken: (endTime ?? 0) - (startTime ?? 0),
      raw: call,
      parentNodes: [...tree.parentNodes, tree],
      isRoot: false,
    });

    return;
  }

  if (!matchingNode) {
    const newNode: StackTreeNode = {
      children: [],
      name,
      methodName,
      path,
      startTime: 0,
      endTime: 0,
      timeTaken: 0,
      nodeId: `${v4()}-${methodName}-${path}`, // Placeholder
      parentNodes: [...tree.parentNodes, tree],
      isRoot: false,
    };

    // otherwise create a new node
    tree.children.push(newNode);
    matchingNode = newNode;
  }

  addCallToTree(matchingNode, call, stack, index + 1);
};

export const createTreeFromCalls = (recordJSON: RecordJSONRaw, appName: string) => {
  const startTime = new Date(recordJSON.perf.start_time).getTime();
  const endTime = new Date(recordJSON.perf.end_time).getTime();

  const tree: StackTreeNode = {
    children: [],
    name: appName,
    startTime,
    endTime,
    timeTaken: endTime - startTime,
    path: '',
    parentNodes: [],
    id: 0,
    isRoot: true,
    nodeId: ROOT_NODE_ID, // ID for the record viewer, since function Ids may not be unique.
    raw: {
      stack: [],
      args: { str_or_query_bundle: '' },
      error: null,
      rets: [],
      perf: recordJSON.perf,
      pid: -1,
      tid: -1,
    },
  };

  recordJSON.calls.forEach((call) => {
    addCallToTree(tree, call, call.stack, 0);
  });

  return tree;
};

export const getSelector = (node: StackTreeNode) => {
  if (!node) return '';

  const components = [`Select.Record`, node.path, node.methodName].filter(Boolean);

  return components.join('.');
};

export const createNodeMap = (node: StackTreeNode) => {
  const result: Record<string, StackTreeNode> = {};

  const queue = [node];

  while (queue.length !== 0) {
    const currNode = queue.pop();

    if (!currNode) continue;

    result[currNode.nodeId] = currNode;

    queue.push(...currNode.children);
  }

  return result;
};
