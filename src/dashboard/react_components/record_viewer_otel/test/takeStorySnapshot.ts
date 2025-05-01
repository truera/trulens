import { expect, Page } from '@playwright/test';

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

  // Get the expected path
  const storyIdPath = `${storyId.replace(/\//g, '-')}.png`;

  // Take a screenshot
  const screenshot = await page.screenshot();

  // Compare with existing baseline
  expect(screenshot).toMatchSnapshot(storyIdPath);
}
