export const uniq = <T>(array: T[]): T[] => {
  return Array.from(new Set(array));
};
