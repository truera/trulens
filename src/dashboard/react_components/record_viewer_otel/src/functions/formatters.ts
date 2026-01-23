const { languages } = navigator;

const languagesToUse = [...languages];

export const formatNumber = (value: number, options: Intl.NumberFormatOptions): string => {
  return new Intl.NumberFormat(languagesToUse, options).format(value);
};

/**
 * Formatting timestamp to display
 *
 * @param timestampInSeconds - timestamp in seconds
 * @returns Human-readable formatted timestamp string
 */
export const formatTime = (timestampInSeconds: number) => {
  if (timestampInSeconds === undefined || timestampInSeconds === null) return '';

  const jsDate = new Date(timestampInSeconds * 1000);

  const formatter = new Intl.DateTimeFormat(languagesToUse, {
    weekday: 'long',
    year: 'numeric',
    month: 'numeric',
    day: 'numeric',
    hour: 'numeric',
    minute: 'numeric',
    second: 'numeric',
    fractionalSecondDigits: 3,
    hour12: true,
    timeZoneName: 'short',
  });

  return formatter.format(jsDate);
};

/**
 * Formatting duration to display.
 *
 * @param durationInSeconds - duration in seconds
 * @returns Human-readable formatted timestamp duration string
 */
export const formatDuration = (durationInSeconds: number) => {
  if (durationInSeconds === null || durationInSeconds === undefined) return '';

  const { format } = new Intl.NumberFormat(languagesToUse, {
    minimumFractionDigits: 0,
    maximumFractionDigits: 3,
  });

  if (durationInSeconds < 0.001) return `${format(durationInSeconds * 1000_000)} µs`;
  if (durationInSeconds < 1) return `${format(durationInSeconds * 1000)} ms`;

  return `${format(durationInSeconds)} s`;
};
