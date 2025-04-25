import { CallJSONRaw } from '@/utils/types';

export const ROOT_NODE_ID = 'root-root-root';

export class StackTreeNode {
  children: StackTreeNode[];

  name: string;

  id: string;

  startTime = 0;

  endTime = 0;

  raw?: CallJSONRaw;

  parentId: string;

  constructor({
    children = [],
    name,
    id,
    startTime,
    endTime,
    raw,
    parentId,
  }: {
    children?: StackTreeNode[];
    name: string;
    id: string;
    raw: CallJSONRaw;
    parentId: string;
    startTime: number;
    endTime: number;
  }) {
    this.startTime = startTime;
    this.endTime = endTime;
    this.children = children;
    this.name = name;
    this.raw = raw;
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
