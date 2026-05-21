import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: '.',
  timeout: 15000,
  expect: { timeout: 4000 },
  workers: 1,
  fullyParallel: false,
  reporter: [['json']],
  use: {
    headless: true,
    actionTimeout: 4000,
    navigationTimeout: 6000,
  },
});
