import { useColorScheme } from '@mui/material';
import { createContext, ReactElement, useEffect } from 'react';
import { Theme as StreamlitTheme } from 'streamlit-component-lib';

interface ThemeProviderProps {
  children: ReactElement;
  streamlitTheme?: StreamlitTheme;
}

export const DEFAULT_THEME: StreamlitTheme = {
  backgroundColor: '#ffffff',
  base: 'light',
  font: '"Source Sans Pro", sans-serif',
  primaryColor: '#ff4b4b',
  secondaryBackgroundColor: '#f0f2f6',
  textColor: '#31333F',
};

export const StreamlitThemeContext = createContext(DEFAULT_THEME);

export function ThemeProvider(props: ThemeProviderProps) {
  const { children, streamlitTheme = DEFAULT_THEME } = props;
  const { setMode } = useColorScheme();

  useEffect(() => {
    setMode(streamlitTheme.base === 'light' ? 'light' : 'dark');
  }, [streamlitTheme.base, setMode]);

  return children;
}
