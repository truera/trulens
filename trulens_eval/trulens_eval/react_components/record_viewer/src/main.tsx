import './assets/fonts.css';

import { StyledEngineProvider, ThemeProvider } from '@mui/material';
import React from 'react';
import ReactDOM from 'react-dom/client';

import RecordViewer from './RecordViewer';
import TrueraTheme from './utils/TrueraTheme';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <StyledEngineProvider injectFirst>
      <ThemeProvider theme={TrueraTheme}>
        <RecordViewer />
      </ThemeProvider>
    </StyledEngineProvider>
  </React.StrictMode>
);
