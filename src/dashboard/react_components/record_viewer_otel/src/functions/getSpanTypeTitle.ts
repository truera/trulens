export const getSpanTypeTitle = (spanType: string) => {
  if (!spanType) return 'Unknown';

  const splitSpanType = spanType.split('_');

  splitSpanType[0] = splitSpanType[0][0].toUpperCase() + splitSpanType[0].slice(1);

  return splitSpanType.join(' ');
};
