import { createTheme, Theme } from '@mui/material';
import grey from '@mui/material/colors/grey';

import * as Colors from './colors';

declare module '@mui/material/styles' {
  interface TypographyVariants {
    menu: React.CSSProperties;
    bodyStrong: React.CSSProperties;
    code: React.CSSProperties;
  }

  // allow configuration using `createTheme`
  interface TypographyVariantsOptions {
    menu?: React.CSSProperties;
    bodyStrong?: React.CSSProperties;
    code?: React.CSSProperties;
  }

  interface Palette {
    important: Palette['primary'];
    destructive: Palette['primary'];
  }
  interface PaletteOptions {
    important: PaletteOptions['primary'];
    destructive: PaletteOptions['primary'];
  }

  interface PaletteColor {
    lighter?: string;
  }

  interface SimplePaletteColorOptions {
    lighter?: string;
  }
}

// Update the Typography's variant prop options
declare module '@mui/material/Typography' {
  interface TypographyPropsVariantOverrides {
    menu: true;
    bodyStrong: true;
    code: true;
  }
}
// Update the Typography's variant prop options
declare module '@mui/material/Typography' {
  interface TypographyPropsVariantOverrides {
    poster: true;
    code: true;
    h3: false;
  }
}

declare module '@mui/material/Button' {
  interface ButtonPropsColorOverrides {
    important: true;
    destructive: true;
  }
}

const fontFamily = ['SourceSansPro', 'Arial', 'sans-serif'].join(',');

const containerStyle = {
  WebkitFontSmoothing: 'auto',
  height: '100%',
  width: '100%',
  margin: 0,
  fontFamily,
};

const TrueraTheme: Theme = createTheme({
  palette: {
    primary: {
      lighter: Colors.PRIMARY_COLOR_LIGHTEST,
      light: Colors.PRIMARY_COLOR_LIGHT,
      main: Colors.PRIMARY_COLOR,
      dark: Colors.PRIMARY_COLOR_DARKEST,
    },
    info: {
      light: Colors.INFO_LIGHT,
      main: Colors.INFO,
      dark: Colors.INFO_DARK,
    },
    action: {
      hover: Colors.PRIMARY_COLOR_LIGHTEST,
      hoverOpacity: 0.25,
    },
    error: {
      light: Colors.FOCUS_SALMON,
      main: Colors.RED,
    },
    grey: {
      50: Colors.HOVER_GRAY,
      100: Colors.BACKGROUND_GRAY,
      300: Colors.GRAY,
      500: Colors.DISABLED_TEXT_GRAY,
      600: Colors.DARK_GRAY,
      900: Colors.BLACK,
    },
    important: {
      main: Colors.FOCUS_YELLOW,
    },
    destructive: {
      main: Colors.ALERT_RED,
    },
  },
  typography: {
    // 1rem = 16px
    fontFamily,
    // Page-title
    h1: {
      fontSize: '2rem',
      fontWeight: 600,
      lineHeight: 1.2, // 120%
      letterSpacing: '-0.02em',
      margin: 0,
    },
    // Widget Title
    h2: {
      fontSize: '1.5rem',
      fontWeight: 600,
      lineHeight: 1.35, // 135%
      letterSpacing: '-0.02em',
      margin: 0,
    },
    // Big number
    h3: {
      fontSize: '1.5rem',
      fontWeight: 400,
      lineHeight: 1.35, // 135%
      margin: 0,
    },
    // Header
    h4: {
      fontSize: '1.25rem',
      fontWeight: 600,
      lineHeight: 1.2, // 120%
      margin: 0,
    },
    h5: {
      fontSize: '1.1rem',
      fontWeight: 600,
      lineHeight: 1.1,
      margin: 0,
    },
    // Standard text, default in <body>
    body2: {
      fontSize: '1rem',
      fontWeight: 400,
      lineHeight: 1.5, // 150%
      letterSpacing: '0.01em',
      margin: 0,
    },
    bodyStrong: {
      fontSize: '1rem',
      fontWeight: 600, // bold
      lineHeight: 1.5, // 150%
      letterSpacing: '0.01em',
      margin: 0,
      color: grey[600],
    },
    // Bold text <b>
    fontWeightBold: 600,
    // Button
    button: {
      fontSize: '0.875rem',
      fontWeight: 600,
      lineHeight: 1.15, // 115%
      letterSpacing: '0.03em',
      textTransform: 'uppercase',
    },
    // Detail text
    subtitle1: {
      fontSize: '0.75rem',
      fontWeight: 400,
      lineHeight: 1.3, // 130%
      letterSpacing: '0.01em',
      color: grey[600],
    },
    // Text style for navigation components
    menu: {
      fontWeight: 600,
      fontSize: '0.875rem',
      lineHeight: 1.15,
      letterSpacing: '0.03em',
    },
    code: {
      color: 'rgb(9,171,59)',
      fontFamily: '"Source Code Pro", monospace',
      margin: 0,
      fontSize: '0.75em',
      borderRadius: '0.25rem',
      background: 'rgb(250,250,250)',
      width: 'fit-content',
    },
  },
});

TrueraTheme.components = {
  MuiCssBaseline: {
    styleOverrides: {
      html: containerStyle,
      body: containerStyle,
      '#root': containerStyle,
      h1: TrueraTheme.typography.h1,
      h2: TrueraTheme.typography.h2,
      h3: TrueraTheme.typography.h3,
      h4: TrueraTheme.typography.h4,
      h5: TrueraTheme.typography.h5,
      p: TrueraTheme.typography.body2,
      '.link': {
        color: TrueraTheme.palette.primary.main,
        textDecoration: 'underline',
        cursor: 'pointer',
      },
      '.disabled': {
        color: TrueraTheme.palette.grey[400],
      },
      '.input': {
        color: TrueraTheme.palette.grey[600],
        fontStyle: 'italic',
      },
      '.detail': TrueraTheme.typography.subtitle1,
      '.dot': {
        height: TrueraTheme.spacing(2),
        width: TrueraTheme.spacing(2),
        borderRadius: TrueraTheme.spacing(2),
        marginRight: TrueraTheme.spacing(1),
        display: 'flex',
      },
      a: {
        color: 'unset',
        '&:link': {
          textDecoration: 'none',
        },
        '&:visited': {
          textDecoration: 'none',
        },
      },
    },
  },
  MuiButton: {
    styleOverrides: {
      sizeLarge: {
        padding: TrueraTheme.spacing(2),
        height: TrueraTheme.spacing(6),
      },
      // Medium styles retired for now, but since they are the default in MUI,
      // turn them into large.
      sizeMedium: {
        padding: TrueraTheme.spacing(2),
        height: TrueraTheme.spacing(6),
      },
      sizeSmall: {
        height: TrueraTheme.spacing(4),
        lineHeight: 1,
      },
    },
    variants: [
      {
        props: { color: 'primary', variant: 'contained' },
        style: {
          ':hover': {
            backgroundColor: Colors.PRIMARY_COLOR_DARK,
          },
        },
      },
      {
        props: { color: 'primary', variant: 'outlined' },
        style: {
          borderColor: Colors.PRIMARY_COLOR_LIGHT,
          ':hover': {
            borderColor: Colors.PRIMARY_COLOR_LIGHT,
            backgroundColor: Colors.PRIMARY_COLOR_LIGHTEST,
          },
        },
      },
      {
        props: { color: 'important' },
        style: {
          color: TrueraTheme.palette.grey[900],
          ':hover': {
            backgroundColor: Colors.WARNING,
          },
        },
      },
      {
        props: { color: 'destructive' },
        style: {
          color: '#FFFFFF',
          ':hover': {
            background: Colors.DARK_RED,
          },
        },
      },
    ],
  },
  MuiInputBase: {
    styleOverrides: {
      root: {
        height: TrueraTheme.spacing(5),
      },
    },
  },
  MuiTouchRipple: {
    styleOverrides: {
      root: {
        height: TrueraTheme.spacing(6),
      },
    },
  },
};

export default TrueraTheme;
