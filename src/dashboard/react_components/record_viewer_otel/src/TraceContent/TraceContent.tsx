import JSONViewer from '@/JSONViewer';
import { Typography } from '@mui/material';

export interface TraceContentProps {
  rawValue: unknown;
}

const MAX_STRING_LENGTH = 280;

export const TraceContent = (props: TraceContentProps) => {
  const { rawValue } = props;

  if (!rawValue) {
    return <Typography>No information found.</Typography>;
  }

  let value = rawValue;

  if (typeof rawValue === 'string') {
    try {
      value = JSON.parse(rawValue);
    } catch {
      // Do nothing if it fails to parse as JSON.
    }
  }

  if (
    Array.isArray(value) &&
    value.every((element) => typeof element === 'string') &&
    value.some((element) => element.length >= MAX_STRING_LENGTH)
  ) {
    return <JSONViewer src={value} />;
  }

  if (typeof value === 'object') {
    return <JSONViewer src={rawValue} />;
  }

  return <Typography>{String(value) || 'No information found.'}</Typography>;
};
