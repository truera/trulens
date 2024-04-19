import { StackTreeNode } from './StackTreeNode';
import { CallJSONRaw, PerfJSONRaw, RecordJSONRaw, StackJSONRaw } from './types';

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
  const name = getClassNameFromCell(stackCell);

  // Given a recorded call, see if its parent already exist as a child of the tree.
  let matchingNode = tree.children.find(
    (node) =>
      node.name === name &&
      node.startTime <= new Date(call.perf.start_time).getTime() &&
      (!node.endTime || node.endTime >= new Date(call.perf.end_time).getTime())
  );

  // if we are currently at the top most cell of the stack...
  if (index === stack.length - 1) {
    const { startTime, endTime } = getStartAndEndTimes(call.perf);

    // ...and there is a matching node, then this call must be for this node. Update
    // the start/end time, and raw call correspondingly.
    if (matchingNode) {
      matchingNode.startTime = startTime;
      matchingNode.endTime = endTime;
      matchingNode.raw = call;

      return;
    }

    // Otherwise this is a new call that is unrecorded, add it in
    tree.children.push(
      new StackTreeNode({
        name,
        raw: call,
        parentNodes: [...tree.parentNodes, tree],
        perf: call.perf,
        stackCell,
      })
    );

    return;
  }

  // No matching node, so this is a new path. Create a new node for it.
  if (!matchingNode) {
    const newNode = new StackTreeNode({
      name,
      stackCell,
      parentNodes: [...tree.parentNodes, tree],
    });

    tree.children.push(newNode);
    matchingNode = newNode;
  }

  addCallToTree(matchingNode, call, stack, index + 1);
};

export const createTreeFromCalls = (recordJSON: RecordJSONRaw, appName: string) => {
  const tree = new StackTreeNode({
    name: appName,
    perf: recordJSON.perf,
  });

  recordJSON.calls.forEach((call) => {
    addCallToTree(tree, call, call.stack, 0);
  });

  return tree;
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
