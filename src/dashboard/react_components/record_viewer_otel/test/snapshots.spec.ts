import { test } from '@playwright/test';
import { takeStorySnapshot } from './takeStorySnapshot';
import { readFileSync } from 'fs';
import * as path from 'path';

// Read stories outside the test to make them available to all tests
const storybookDir = path.resolve(process.cwd(), 'storybook-static');
const indexPath = path.join(storybookDir, 'index.json');
let storyIds: string[] = [];

try {
  const indexContent = readFileSync(indexPath, 'utf-8');
  const storybookData = JSON.parse(indexContent);

  // Extract story IDs based on the structure of index.json
  if (storybookData.entries) {
    storyIds = Object.keys(storybookData.entries);
  } else if (storybookData.stories) {
    storyIds = Object.keys(storybookData.stories);
  }
} catch (error) {
  console.error('Failed to read storybook index:', error);
  throw error;
}

test.describe('Storybook Visual Tests', () => {
  // Generate a test for each story
  for (const storyId of storyIds) {
    test(`Snapshot test: ${storyId}`, async ({ page }) => {
      await takeStorySnapshot(page, storyId);
    });
  }
});
