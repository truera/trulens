import { Box, SxProps, Theme, Typography } from '@mui/material';
import { PropsWithChildren, useId } from 'react';

type PanelProps = PropsWithChildren<{ header: string; id?: string }>;

export default function Panel({ header, children, id }: PanelProps) {
  // Generate a unique ID if one isn't provided
  const reactId = useId();
  const panelId = id ?? reactId;
  const headerId = `${panelId}-header`;
  const contentId = `${panelId}-content`;

  return (
    <Box sx={panelSx} role="section" aria-labelledby={headerId}>
      <Box className="panel-header" component="header" id={headerId}>
        <Typography variant="body2" fontWeight="bold" component="h3">
          {header}
        </Typography>
      </Box>

      <Box className="panel-content" id={contentId} aria-labelledby={headerId}>
        {children}
      </Box>
    </Box>
  );
}

const panelSx: SxProps<Theme> = ({ spacing, palette }) => ({
  borderRadius: spacing(0.5),
  border: `1px solid ${palette.grey[300]}`,
  width: '100%',

  '& .panel-header': {
    background: palette.grey[50],
    p: 1,
    borderBottom: `1px solid ${palette.grey[300]}`,
  },

  '& .panel-content': {
    p: 2,
  },
});
