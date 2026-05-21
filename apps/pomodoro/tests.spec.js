import { test, expect } from '@playwright/test';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const candidate = process.env.CANDIDATE_HTML;
if (!candidate) throw new Error('CANDIDATE_HTML env var required');
const APP_URL = 'file://' + path.resolve(candidate);

const display  = '[data-testid="timer-display"]';
const phase    = '[data-testid="phase-display"]';
const cycle    = '[data-testid="cycle-count"]';

test.beforeEach(async ({ page }) => {
  // 仮想時計を有効化（候補 HTML 読み込み前に install）
  await page.clock.install({ time: new Date('2026-05-20T00:00:00') });
  await page.goto(APP_URL);
  await page.evaluate(() => localStorage.clear());
  await page.reload();
});

test('T1: initial timer display is 25:00', async ({ page }) => {
  const text = await page.locator(display).textContent();
  expect(text.trim()).toMatch(/^0?25:00$/);
});

test('T2: initial phase is work', async ({ page }) => {
  await expect(page.locator(phase)).toContainText(/work/i);
});

test('T3: initial cycle count is 0', async ({ page }) => {
  await expect(page.locator(cycle)).toContainText('0');
});

test('T4: start button starts the timer', async ({ page }) => {
  await page.click('[data-testid="btn-start"]');
  await page.clock.runFor(2000); // 2 秒進める
  const text = await page.locator(display).textContent();
  // 25:00 ではなく < 25:00 になっているはず
  expect(text.trim()).not.toMatch(/^0?25:00$/);
});

test('T5: pause button stops the timer', async ({ page }) => {
  await page.click('[data-testid="btn-start"]');
  await page.clock.runFor(3000);
  await page.click('[data-testid="btn-pause"]');
  const t1 = (await page.locator(display).textContent()).trim();
  await page.clock.runFor(5000);
  const t2 = (await page.locator(display).textContent()).trim();
  expect(t2).toBe(t1);
});

test('T6: reset button returns timer to start of phase', async ({ page }) => {
  await page.click('[data-testid="btn-start"]');
  await page.clock.runFor(5000);
  await page.click('[data-testid="btn-reset"]');
  const t = (await page.locator(display).textContent()).trim();
  expect(t).toMatch(/^0?25:00$/);
});

test('T7: skip button transitions work to break', async ({ page }) => {
  await page.click('[data-testid="btn-skip"]');
  await expect(page.locator(phase)).toContainText(/break/i);
});

test('T8: skip from break returns to work and increments cycle', async ({ page }) => {
  await page.click('[data-testid="btn-skip"]'); // work -> break
  await page.click('[data-testid="btn-skip"]'); // break -> work
  await expect(page.locator(phase)).toContainText(/work/i);
  await expect(page.locator(cycle)).toContainText('1');
});

test('T9: work auto-transitions to break when timer hits 0', async ({ page }) => {
  // work を 1 分に短縮して時計前進量を抑える (Playwright タイマー budget 回避)
  await page.fill('[data-testid="setting-work"]', '1');
  await page.click('[data-testid="btn-save-settings"]');
  await page.click('[data-testid="btn-reset"]');
  await page.click('[data-testid="btn-start"]');
  await page.clock.runFor(60 * 1000 + 1000); // 1 min + 1s
  await expect(page.locator(phase)).toContainText(/break/i);
});

test('T10: settings change work duration', async ({ page }) => {
  await page.fill('[data-testid="setting-work"]', '10');
  await page.click('[data-testid="btn-save-settings"]');
  await page.click('[data-testid="btn-reset"]');
  const t = (await page.locator(display).textContent()).trim();
  expect(t).toMatch(/^0?10:00$/);
});

test('T11: settings persist across reload', async ({ page }) => {
  await page.fill('[data-testid="setting-work"]', '15');
  await page.click('[data-testid="btn-save-settings"]');
  await page.reload();
  const val = await page.locator('[data-testid="setting-work"]').inputValue();
  expect(val).toBe('15');
});

test('T12: cycle count persists across reload', async ({ page }) => {
  await page.click('[data-testid="btn-skip"]'); // work -> break
  await page.click('[data-testid="btn-skip"]'); // break -> work (cycle 1)
  await page.reload();
  await expect(page.locator(cycle)).toContainText('1');
});
