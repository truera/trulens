import { StackTreeNode } from '@/utils/StackTreeNode';
import { CallJSONRaw, RecordJSONRaw, StackJSONRaw } from '@/utils/types';

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

const addCallToTree = (tree: StackTreeNode, call: CallJSONRaw, stack: StackJSONRaw[], index: number) => {
  const stackCell = stack[index];
  const name = getClassNameFromCell(stackCell);

  // Given a recorded call, see if its parent already exist as a child of the tree.
  let matchingNode = tree.children.find(
    (node) =>
      node.name === name &&
      // Using string comparisons because Javascript Date doesn't compare microseconds.
      (!node.startTime || node.startTime <= getMicroseconds(call.perf?.start_time)) &&
      (!node.endTime || node.endTime >= getMicroseconds(call.perf?.end_time))
  );

  // if we are currently at the top most cell of the stack...
  if (index === stack.length - 1) {
    // ...and there is a matching node, then this call must be for this node. Update
    // the start/end time, and raw call correspondingly.
    if (matchingNode) {
      matchingNode.startTime = getMicroseconds(call.perf?.start_time);
      matchingNode.endTime = getMicroseconds(call.perf?.end_time);
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

/**
 * Formatting timestamp to display
 *
 * @param timestampInMicroSeconds - timestamp in microseconds
 * @returns Human-readable formatted timestamp string
 */
export const formatTime = (timestampInMicroSeconds: number) => {
  if (!timestampInMicroSeconds) return '';

  const jsDate = new Date(timestampInMicroSeconds / 1000);

  return `${jsDate.toLocaleDateString()} ${jsDate.toLocaleTimeString('en-US', {
    hour12: false,
  })}.${timestampInMicroSeconds % 1_000_000}`;
};

/**
 * Formatting duration to display.
 *
 * @param durationInMicroSeconds - duration in microseconds
 * @returns Human-readable formatted timestamp duration string
 */
export const formatDuration = (durationInMicroSeconds: number) => {
  if (Number.isNaN(durationInMicroSeconds)) return '';

  if (durationInMicroSeconds < 1000) return `${durationInMicroSeconds} µs`;
  if (durationInMicroSeconds < 1_000_000) return `${Math.round(durationInMicroSeconds / 1000)} ms`;

  return `${Math.round(durationInMicroSeconds / 1_000_000_000)} s`;
};

export const getMicroseconds = (timestamp: string) => {
  if (!timestamp) return 0;

  const jsTimestampInMs = new Date(timestamp).valueOf();
  let jsTimestampInMicroS = jsTimestampInMs * 1000;

  const splitTimestamp = timestamp.split('.');
  if (splitTimestamp.length === 2 && splitTimestamp[1].length === 6) {
    const microseconds = Number(splitTimestamp[1].substring(3));

    if (!Number.isNaN(microseconds)) {
      jsTimestampInMicroS += microseconds;
    }
  }

  return jsTimestampInMicroS;
};
