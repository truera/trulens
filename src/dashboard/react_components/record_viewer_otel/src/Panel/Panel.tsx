import { Box, SxProps, Theme, Typography } from '@mui/material';
import { PropsWithChildren, useId, useState } from 'react';
import KeyboardArrowDownRounded from '@mui/icons-material/KeyboardArrowDownRounded';
import KeyboardArrowUpRounded from '@mui/icons-material/KeyboardArrowUpRounded';

type PanelProps = PropsWithChildren<{
  header: string;
  id?: string;
  expanded?: boolean;
}>;

export default function Panel({ header, children, id, expanded: initialExpanded = true }: PanelProps) {
  // Handle accordion state - use the prop for initial value
  const [expanded, setExpanded] = useState(initialExpanded);
  const toggleExpand = () => {
    setExpanded(!expanded);
  }
  // Generate a unique ID if one isn't provided
  const reactId = useId();
  const panelId = id ?? reactId;
  const headerId = `${panelId}-header`;
  const contentId = `${panelId}-content`;

  return (
    <Box sx={panelSx} role="section" aria-labelledby={headerId}>
      <Box
        className="panel-header"
        component="header"
        id={headerId}
        sx={{ display: 'flex', alignItems: 'center' }}
        aria-expanded={expanded}
        aria-controls={contentId}
        onClick={toggleExpand}
      >
        {expanded ? <KeyboardArrowUpRounded /> : <KeyboardArrowDownRounded />}
        <Typography variant="body2" fontWeight="bold" component="h3">
          {header}
        </Typography>
      </Box>
      {expanded && (
        <Box className="panel-content" id={contentId} aria-labelledby={headerId} aria-hidden={!expanded}>
          {children}
        </Box>
      )}
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

  '& .panel-accordion-button': {
    background: 'transparent',
    border: 'none',
    cursor: 'pointer',
    marginRight: spacing(1),
  },

  '& .panel-content': {
    p: 2,
  },
});
