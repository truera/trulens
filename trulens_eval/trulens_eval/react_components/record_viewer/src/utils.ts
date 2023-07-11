import { CallJSONRaw, PerfJSONRaw, RecordJSONRaw, StackJSONRaw, StackTreeNode } from './types';

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
  return stackCell.path.path
    .map((p) => p?.item_or_attribute)
    .filter(Boolean)
    .join('.');
};

/**
 * Gets the start and end times based on the performance
 * data provided.
 *
 * @param perf - PerfJSONRaw The performance data to be analyzed
 * @returns an object containing the start and end times based on the performance
 * data provided
 */
export const getStartAndEndTimes = (perf: PerfJSONRaw) => {
  return {
    startTime: perf?.start_time ? new Date(perf.start_time) : undefined,
    endTime: perf?.end_time ? new Date(perf.end_time) : undefined,
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
      (node.startTime ?? 0) <= new Date(call.perf.start_time) &&
      (node.endTime ?? Infinity) >= new Date(call.perf.end_time)
  );

  // if we are currently at the top most cell of the stack
  if (index === stack.length - 1) {
    const { startTime, endTime } = getStartAndEndTimes(call.perf);

    if (matchingNode) {
      matchingNode.startTime = startTime;
      matchingNode.endTime = endTime;
      matchingNode.raw = call;

      return;
    }

    tree.children.push({
      children: undefined,
      name: getClassNameFromCell(stackCell),
      path: getPathName(stackCell),
      methodName: getMethodNameFromCell(stackCell),
      startTime,
      endTime,
      raw: call,
    });

    return;
  }

  if (!matchingNode) {
    const newNode = {
      children: [],
      name: getClassNameFromCell(stackCell),
      methodName: getMethodNameFromCell(stackCell),
      path: getPathName(stackCell),
    };

    // otherwise create a new node
    tree.children.push(newNode);
    matchingNode = newNode;
  }

  addCallToTree(matchingNode, call, stack, index + 1);
};

export const createTreeFromCalls = (recordJSON: RecordJSONRaw) => {
  const tree: StackTreeNode = {
    children: [],
    name: 'App',
    startTime: new Date(recordJSON.perf.start_time),
    endTime: new Date(recordJSON.perf.end_time),
  };

  recordJSON.calls.forEach((call) => {
    addCallToTree(tree, call, call.stack, 0);
  });

  return tree;
};
