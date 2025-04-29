import { Box, SxProps, Theme, Typography } from '@mui/material';
import { PropsWithChildren } from 'react';

type PanelProps = PropsWithChildren<{ header: string }>;

export default function Panel({ header, children }: PanelProps) {
  return (
    <Box sx={panelSx}>
      <Box className="panel-header">
        <Typography variant="body2" fontWeight="bold">
          {header}
        </Typography>
      </Box>

      <Box className="panel-content">{children}</Box>
    </Box>
  );
}

const panelSx: SxProps<Theme> = ({ spacing, vars }) => ({
  borderRadius: spacing(0.5),
  border: `1px solid ${vars.palette.grey[300]}`,
  width: '100%',

  '& .panel-header': {
    background: vars.palette.grey[50],
    p: 1,
    borderBottom: `1px solid ${vars.palette.grey[300]}`,
  },

  '& .panel-content': {
    p: 2,
  },
});
