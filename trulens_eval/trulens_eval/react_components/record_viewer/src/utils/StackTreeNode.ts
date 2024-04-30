import { Span } from '@/utils/Span';
import { CallJSONRaw, PerfJSONRaw, StackJSONRaw } from '@/utils/types';
import { getMethodNameFromCell, getPathName } from '@/utils/utils';

export const ROOT_NODE_ID = 'root-root-root';

export class StackTreeNode<SpanType extends Span = Span> {
  children: StackTreeNode[];

  name: string;

  path = '';

  methodName = '';

  startTime = 0;

  endTime = 0;

  raw?: CallJSONRaw;

  span?: SpanType;

  parentNodes: StackTreeNode[] = [];

  constructor({
    children = [],
    name,
    stackCell,
    perf,
    raw,
    parentNodes = [],
    span,
  }: {
    children?: StackTreeNode[];
    name: string;
    stackCell?: StackJSONRaw;
    raw?: CallJSONRaw;
    parentNodes?: StackTreeNode[];
    perf?: PerfJSONRaw;
    span?: SpanType;
  }) {
    if (perf) {
      const startTime = new Date(perf.start_time).getTime();
      const endTime = new Date(perf.end_time).getTime();
      this.startTime = startTime;
      this.endTime = endTime;
    }

    this.children = children;
    this.name = name;
    this.raw = raw;
    this.parentNodes = parentNodes;
    this.span = span;

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
