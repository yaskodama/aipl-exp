import { test, expect } from '@playwright/test';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const candidate = process.env.CANDIDATE_HTML;
if (!candidate) throw new Error('CANDIDATE_HTML env var required');
const APP_URL = 'file://' + path.resolve(candidate);

const list  = '[data-testid="todo-list"]';
const items = '[data-testid="todo-item"]';
const input = '[data-testid="new-todo-input"]';

async function addItem(page, text, useButton = false) {
  await page.fill(input, text);
  if (useButton) await page.click('[data-testid="add-btn"]');
  else           await page.press(input, 'Enter');
}

test.beforeEach(async ({ page }) => {
  await page.goto(APP_URL);
  await page.evaluate(() => localStorage.clear());
  await page.reload();
});

test('T1: empty list on load', async ({ page }) => {
  await expect(page.locator(items)).toHaveCount(0);
});

test('T2: add item via Enter', async ({ page }) => {
  await addItem(page, '牛乳を買う');
  await expect(page.locator(items)).toHaveCount(1);
});

test('T3: add item via button', async ({ page }) => {
  await addItem(page, '宿題する', true);
  await expect(page.locator(items)).toHaveCount(1);
});

test('T4: item shows the text', async ({ page }) => {
  await addItem(page, 'ABC');
  await expect(page.locator('[data-testid="todo-text"]').first()).toContainText('ABC');
});

test('T5: delete an item', async ({ page }) => {
  await addItem(page, 'delete me');
  await page.click('[data-testid="todo-delete"]');
  await expect(page.locator(items)).toHaveCount(0);
});

test('T6: toggle completion', async ({ page }) => {
  await addItem(page, 'task');
  const toggle = page.locator('[data-testid="todo-toggle"]').first();
  await toggle.check();
  await expect(toggle).toBeChecked();
});

test('T7: completed has line-through style', async ({ page }) => {
  await addItem(page, 'styled');
  await page.locator('[data-testid="todo-toggle"]').first().check();
  const deco = await page.locator('[data-testid="todo-text"]').first()
    .evaluate(el => getComputedStyle(el).textDecorationLine);
  expect(deco).toContain('line-through');
});

test('T8: filter active hides completed', async ({ page }) => {
  await addItem(page, 'A'); await addItem(page, 'B');
  await page.locator('[data-testid="todo-toggle"]').first().check();
  await page.click('[data-testid="filter-active"]');
  const visible = await page.locator(items).evaluateAll(
    els => els.filter(e => e.offsetParent !== null).length
  );
  expect(visible).toBe(1);
});

test('T9: filter completed hides active', async ({ page }) => {
  await addItem(page, 'A'); await addItem(page, 'B');
  await page.locator('[data-testid="todo-toggle"]').first().check();
  await page.click('[data-testid="filter-completed"]');
  const visible = await page.locator(items).evaluateAll(
    els => els.filter(e => e.offsetParent !== null).length
  );
  expect(visible).toBe(1);
});

test('T10: filter all shows all', async ({ page }) => {
  await addItem(page, 'A'); await addItem(page, 'B');
  await page.locator('[data-testid="todo-toggle"]').first().check();
  await page.click('[data-testid="filter-completed"]');
  await page.click('[data-testid="filter-all"]');
  const visible = await page.locator(items).evaluateAll(
    els => els.filter(e => e.offsetParent !== null).length
  );
  expect(visible).toBe(2);
});

test('T11: persistence across reload', async ({ page }) => {
  await addItem(page, 'persist me');
  await page.reload();
  await expect(page.locator(items)).toHaveCount(1);
  await expect(page.locator('[data-testid="todo-text"]').first()).toContainText('persist me');
});

test('T12: clear-completed removes completed only', async ({ page }) => {
  await addItem(page, 'A'); await addItem(page, 'B'); await addItem(page, 'C');
  await page.locator('[data-testid="todo-toggle"]').nth(0).check();
  await page.locator('[data-testid="todo-toggle"]').nth(2).check();
  await page.click('[data-testid="clear-completed"]');
  await expect(page.locator(items)).toHaveCount(1);
  await expect(page.locator('[data-testid="todo-text"]').first()).toContainText('B');
});
