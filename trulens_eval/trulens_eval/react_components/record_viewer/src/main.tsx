import React from 'react';
import { StyledEngineProvider, ThemeProvider } from '@mui/material';
import ReactDOM from 'react-dom/client';
import './assets/fonts.css';
import TrueraTheme from './utils/TrueraTheme';
import RecordViewer from './RecordViewer';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <StyledEngineProvider injectFirst>
      <ThemeProvider theme={TrueraTheme}>
        <RecordViewer />
      </ThemeProvider>
    </StyledEngineProvider>
  </React.StrictMode>
);
