import { createTheme, PaletteOptions, Theme } from '@mui/material';
import type {} from '@mui/material/themeCssVarsAugmentation';

import * as Colors from '@/utils/colors';

declare module '@mui/material/styles' {
  interface TypographyVariants {
    code: React.CSSProperties;
  }

  interface TypographyVariantsOptions {
    code?: React.CSSProperties;
  }
}

// Update the Typography's variant prop options
declare module '@mui/material/Typography' {
  interface TypographyPropsVariantOverrides {
    code: true;
  }
}

const lightPalette: Partial<PaletteOptions> = {
  primary: {
    light: Colors.PRIMARY_COLOR_LIGHT,
    main: Colors.PRIMARY_COLOR,
    dark: Colors.PRIMARY_COLOR_DARKEST,
  },
  grey: {
    50: Colors.BASE_GRAY[0],
    100: Colors.BASE_GRAY[10],
    200: Colors.BASE_GRAY[20],
    300: Colors.BASE_GRAY[30],
    400: Colors.BASE_GRAY[40],
    500: Colors.BASE_GRAY[50],
    600: Colors.BASE_GRAY[60],
    700: Colors.BASE_GRAY[70],
    800: Colors.BASE_GRAY[80],
    900: Colors.BASE_GRAY[90],
  },
  text: {
    primary: Colors.BASE_GRAY[90],
  },
};

const darkPalette: Partial<PaletteOptions> = {
  primary: {
    light: Colors.PRIMARY_COLOR_DARKEST,
    main: Colors.PRIMARY_COLOR_DARK_MODE,
    dark: Colors.PRIMARY_COLOR_LIGHTEST,
  },
  grey: {
    50: Colors.BASE_GRAY[95],
    100: Colors.BASE_GRAY[90],
    200: Colors.BASE_GRAY[80],
    300: Colors.BASE_GRAY[70],
    400: Colors.BASE_GRAY[60],
    500: Colors.BASE_GRAY[50],
    600: Colors.BASE_GRAY[40],
    700: Colors.BASE_GRAY[30],
    800: Colors.BASE_GRAY[20],
    900: Colors.BASE_GRAY[10],
  },
  text: {
    primary: Colors.BASE_GRAY[10],
  },
};

const theme: Theme = createTheme({
  typography: {
    fontFamily: '"Source Sans Pro", sans-serif',
    // Button
    button: {
      fontSize: '0.875rem',
      fontWeight: 600,
      lineHeight: 1.15, // 115%
      letterSpacing: '0.03em',
    },
    // Detail text
    subtitle1: {
      fontSize: '0.75rem',
      fontWeight: 400,
      lineHeight: 1.3, // 130%
      letterSpacing: '0.01em',
    },
    code: {
      color: 'rgb(9,171,59)',
      fontFamily: '"Source Code Pro", monospace',
      margin: 0,
      fontSize: '0.75em',
      borderRadius: '0.25rem',
      width: 'fit-content',
    },
  },
  cssVariables: {
    colorSchemeSelector: '.mode-%s',
  },
  colorSchemes: {
    light: {
      palette: lightPalette,
    },
    dark: {
      palette: darkPalette,
    },
  },
});

export default theme;
