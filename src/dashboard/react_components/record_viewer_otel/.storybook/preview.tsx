import type { Preview, Decorator } from '@storybook/react';
import React from 'react';
import { DEFAULT_THEME, ThemeProvider } from '../src/utils/ThemeProvider';
import { ThemeProvider as MuiThemeProvider, CssBaseline } from '@mui/material';
import theme from '../src/utils/theme';

export const globalTypes = {
  theme: {
    name: 'Theme',
    title: 'Theme',
    description: 'Theme for your components',
    defaultValue: 'light',
    toolbar: {
      icon: 'paintbrush',
      dynamicTitle: true,
      items: [
        { value: 'light', left: '☀️', title: 'Light mode' },
        { value: 'dark', left: '🌙', title: 'Dark mode' },
      ],
    },
  },
};

export const withMuiTheme: Decorator = (Story, context) => {
  return (
    <MuiThemeProvider theme={theme}>
      <CssBaseline />
      <ThemeProvider streamlitTheme={{ ...DEFAULT_THEME, base: context.globals.theme }}>{Story(context)}</ThemeProvider>
    </MuiThemeProvider>
  );
};

const preview: Preview = {
  parameters: {
    // Ensure consistent date/time rendering
    dates: {
      // Set a fixed timezone for consistent rendering across environments
      timezone: 'UTC',
    },
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
    // Control animations
    chromatic: {
      // Disables animations and transitions for more consistent snapshots
      disableSnapshot: false,
      pauseAnimationAtEnd: true,
      delay: 300,
    },
  },

  decorators: [withMuiTheme],
};

export default preview;
