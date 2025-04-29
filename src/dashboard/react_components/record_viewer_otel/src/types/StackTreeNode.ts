import { SpanAttributes } from '@/types/SpanAttributes';

export const ROOT_NODE_ID = 'root-root-root';

export class StackTreeNode {
  children: StackTreeNode[];

  name: string;

  id: string;

  startTime = 0;

  endTime = 0;

  attributes: SpanAttributes;

  parentId: string;

  constructor({
    children = [],
    name,
    id,
    startTime,
    endTime,
    attributes,
    parentId,
  }: {
    children?: StackTreeNode[];
    name: string;
    id: string;
    attributes: SpanAttributes;
    parentId: string;
    startTime: number;
    endTime: number;
  }) {
    this.startTime = startTime;
    this.endTime = endTime;
    this.children = children;
    this.name = name;
    this.attributes = attributes;
    this.parentId = parentId;
    this.id = id;
  }

  get timeTaken() {
    return this.endTime - this.startTime;
  }

  get isRoot() {
    return !this.parentId;
  }

  get label() {
    return this.name.split('.').slice(-2).join('.');
  }
}
