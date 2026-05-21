import { test, expect } from '@playwright/test';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const candidate = process.env.CANDIDATE_HTML;
if (!candidate) throw new Error('CANDIDATE_HTML env var required');
const APP_URL = 'file://' + path.resolve(candidate);

const display = '[data-testid="display"]';
const click = async (page, ...ids) => {
  for (const id of ids) await page.click(`[data-testid="${id}"]`);
};

test.beforeEach(async ({ page }) => {
  await page.goto(APP_URL);
});

test('T1: initial display is 0', async ({ page }) => {
  await expect(page.locator(display)).toHaveText('0');
});

test('T2: pressing 5 shows 5', async ({ page }) => {
  await click(page, 'btn-5');
  await expect(page.locator(display)).toHaveText('5');
});

test('T3: addition 5 + 3 = 8', async ({ page }) => {
  await click(page, 'btn-5', 'btn-add', 'btn-3', 'btn-eq');
  await expect(page.locator(display)).toHaveText('8');
});

test('T4: subtraction 9 - 4 = 5', async ({ page }) => {
  await click(page, 'btn-9', 'btn-sub', 'btn-4', 'btn-eq');
  await expect(page.locator(display)).toHaveText('5');
});

test('T5: multiplication 6 * 7 = 42', async ({ page }) => {
  await click(page, 'btn-6', 'btn-mul', 'btn-7', 'btn-eq');
  await expect(page.locator(display)).toHaveText('42');
});

test('T6: division 8 / 2 = 4', async ({ page }) => {
  await click(page, 'btn-8', 'btn-div', 'btn-2', 'btn-eq');
  await expect(page.locator(display)).toHaveText('4');
});

test('T7: clear resets to 0', async ({ page }) => {
  await click(page, 'btn-7', 'btn-add', 'btn-2', 'btn-clear');
  await expect(page.locator(display)).toHaveText('0');
});

test('T8: decimal 1.5 + 2.5 = 4', async ({ page }) => {
  await click(page, 'btn-1', 'btn-dot', 'btn-5', 'btn-add', 'btn-2', 'btn-dot', 'btn-5', 'btn-eq');
  const text = await page.locator(display).textContent();
  expect(['4', '4.0', '4.00']).toContain(text.trim());
});
