import { CallJSONRaw, PerfJSONRaw, StackJSONRaw } from '@/utils/types';
import { getMethodNameFromCell, getMicroseconds, getPathName } from '@/utils/utils';

export const ROOT_NODE_ID = 'root-root-root';

export class StackTreeNode {
  children: StackTreeNode[];

  name: string;

  path = '';

  methodName = '';

  startTime = 0;

  endTime = 0;

  raw?: CallJSONRaw;

  parentNodes: StackTreeNode[] = [];

  constructor({
    children = [],
    name,
    stackCell,
    perf,
    raw,
    parentNodes = [],
  }: {
    children?: StackTreeNode[];
    name: string;
    stackCell?: StackJSONRaw;
    raw?: CallJSONRaw;
    parentNodes?: StackTreeNode[];
    perf?: PerfJSONRaw;
  }) {
    if (perf) {
      const startTime = getMicroseconds(perf.start_time);
      const endTime = getMicroseconds(perf.end_time);
      this.startTime = startTime;
      this.endTime = endTime;
    }

    this.children = children;
    this.name = name;
    this.raw = raw;
    this.parentNodes = parentNodes;

    if (stackCell) {
      this.path = getPathName(stackCell);
      this.methodName = getMethodNameFromCell(stackCell);
    }
  }

  get timeTaken() {
    return this.endTime - this.startTime;
  }

  get isRoot() {
    return this.parentNodes.length === 0;
  }

  get nodeId() {
    if (this.isRoot) {
      return ROOT_NODE_ID;
    }

    return `${this.methodName}-${this.name}-${this.startTime ?? ''}-${this.endTime ?? ''}`;
  }

  get selector() {
    const components = [`Select.Record`, this.path, this.methodName].filter(Boolean);

    return components.join('.');
  }

  get label() {
    return this.isRoot ? this.name : [this.name, this.methodName].join('.');
  }
}
