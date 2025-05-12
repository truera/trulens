import { Box, Stack, Typography } from '@mui/material';
import { PropsWithChildren } from 'react';

type SectionProps = PropsWithChildren<{
  title: string;
  body?: string;
}>;

export default function Section({ title, body, children }: SectionProps) {
  return (
    <Stack gap={1}>
      <Box sx={{ display: 'flex', gap: 1, alignItems: 'baseline' }}>
        <Typography variant="body2" fontWeight="bold">
          {title}
        </Typography>
      </Box>

      <Typography>{body}</Typography>
      {children}
    </Stack>
  );
}
