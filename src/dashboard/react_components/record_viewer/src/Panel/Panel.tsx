import { Box, SxProps, Theme, Typography } from '@mui/material';
import { PropsWithChildren } from 'react';

type PanelProps = PropsWithChildren<{ header: string }>;

export default function Panel({ header, children }: PanelProps) {
  return (
    <Box sx={panelSx}>
      <Box className="panel-header">
        <Typography variant="body2" fontWeight="bold" color="grey.600">
          {header}
        </Typography>
      </Box>

      <Box className="panel-content">{children}</Box>
    </Box>
  );
}

const panelSx: SxProps<Theme> = ({ spacing, palette }) => ({
  borderRadius: spacing(0.5),
  border: `1px solid ${palette.grey[300]}`,
  width: '100%',

  '& .panel-header': {
    background: palette.grey[100],
    p: 1,
    borderBottom: `1px solid ${palette.grey[300]}`,
  },

  '& .panel-content': {
    p: 2,
  },
});
