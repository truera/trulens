import { getSpanAttributeName } from './getSpanAttributeName';

export const sortSpanKeys = (a: string, b: string) => {
  return getSpanAttributeName(a).localeCompare(getSpanAttributeName(b));
};
