const { languages } = navigator;

export const formatNumber = (value: number, options: Intl.NumberFormatOptions): string => {
  return new Intl.NumberFormat(languages, options).format(value);
};
