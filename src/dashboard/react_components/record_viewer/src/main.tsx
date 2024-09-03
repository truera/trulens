import '@/assets/fonts.css';

import { StyledEngineProvider, ThemeProvider as MuiThemeProvider } from '@mui/material';
import React from 'react';
import ReactDOM from 'react-dom/client';

import RecordViewer from '@/RecordViewer';
import theme from '@/utils/theme';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <StyledEngineProvider injectFirst>
      <MuiThemeProvider theme={theme}>
        <RecordViewer />
      </MuiThemeProvider>
    </StyledEngineProvider>
  </React.StrictMode>
);
