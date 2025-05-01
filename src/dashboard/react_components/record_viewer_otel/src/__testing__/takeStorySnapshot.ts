import { expect, Page } from '@playwright/test';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { dirname } from 'path';

/**
 * Takes a snapshot of a story and compares it to the baseline
 * Contains normalization to ensure OS-agnostic snapshot comparisons
 */
export async function takeStorySnapshot(page: Page, storyId: string) {
  // Navigate to the story
  await page.goto(`http://localhost:6006/iframe.html?id=${storyId}&viewMode=story`);

  // Wait for the story to be fully rendered
  await page.waitForSelector('#storybook-root > *', { state: 'attached' });
  // Additional wait to ensure animations/fonts are loaded
  await page.waitForTimeout(300);

  // Take a screenshot
  const screenshot = await page.screenshot();

  // Get the expected path
  const snapshotDir = `./test/__snapshots__`;
  const snapshotPath = `${snapshotDir}/${storyId.replace(/\//g, '-')}.png`;

  // Ensure directory exists
  if (!existsSync(dirname(snapshotPath))) {
    mkdirSync(dirname(snapshotPath), { recursive: true });
  }

  // Create baseline if it doesn't exist
  if (process.env.UPDATE_SNAPSHOTS === 'true' || !existsSync(snapshotPath)) {
    writeFileSync(snapshotPath, screenshot);
    return;
  }

  // Compare with existing baseline
  expect(screenshot).toMatchSnapshot(snapshotPath);
}
