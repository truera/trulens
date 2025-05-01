import { test } from '@playwright/test';
import { takeStorySnapshot } from '../src/__testing__/takeStorySnapshot';

// Get all story IDs we want to test
const storyIds = [
  'components-tracecontent--empty',
  'components-tracecontent--empty-string',
  'components-tracecontent--simple-string',
  'components-tracecontent--long-string',
  'components-tracecontent--json-string',
  'components-tracecontent--array-of-strings',
  'components-tracecontent--array-of-long-strings',
  'components-tracecontent--simple-object',
  'components-tracecontent--nested-object',
  'components-tracecontent--number-value',
  'components-tracecontent--boolean-value',
  // Add all story IDs you want to test
];

test.describe('Storybook Visual Tests', () => {
  test.beforeAll(async ({ browser }) => {
    // Start Storybook if it's not already running
    // This could be a separate script or you could use a check like:
    // await fetch('http://localhost:6006').catch(() => startStorybook());
  });

  for (const storyId of storyIds) {
    test(`Snapshot for ${storyId}`, async ({ page }) => {
      await takeStorySnapshot(page, storyId);
    });
  }
});
