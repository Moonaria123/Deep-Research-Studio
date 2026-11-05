import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  use: {
    baseURL: "http://127.0.0.1:4174/Deep-Research-Studio/",
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
  },
  webServer: {
    command: "npm run build:demo && vite preview --base /Deep-Research-Studio/ --outDir dist-demo --host 127.0.0.1 --port 4174",
    url: "http://127.0.0.1:4174/Deep-Research-Studio/",
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
