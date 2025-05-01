import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './test',
  timeout: 30000,
  expect: {
    toMatchSnapshot: { threshold: 0.2 }, // Allow slight variations (0.2% difference)
  },
  use: {
    // Browser options
    viewport: { width: 1280, height: 720 },
    // Consistent browser rendering
    deviceScaleFactor: 1,
    // Normalize fonts and rendering
    contextOptions: {
      reducedMotion: 'reduce',
      forcedColors: 'none',
    },
  },
  // Run all tests in a single worker to avoid interference
  workers: 1,
});
