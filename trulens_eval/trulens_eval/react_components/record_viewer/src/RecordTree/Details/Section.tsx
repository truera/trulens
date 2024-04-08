import { Stack, Typography } from '@mui/material';
import { PropsWithChildren } from 'react';

type SectionProps = PropsWithChildren<{
  title: string;
  subtitle?: string;
  body?: string;
}>;

export default function Section({ title, subtitle, body, children }: SectionProps) {
  return (
    <Stack gap={1}>
      <Typography variant="body2" fontWeight="bold">
        {title}
      </Typography>

      <Typography>{body}</Typography>
      {children}
    </Stack>
  );
}
