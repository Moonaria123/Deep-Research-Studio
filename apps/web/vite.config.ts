import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { readFileSync } from "node:fs";

const API_TARGET = process.env.DEEPRESEARCH_API_URL || "http://127.0.0.1:8787";
const licenceFiles = ["LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"].map((fileName) => ({
  fileName,
  source: readFileSync(path.resolve(__dirname, "../..", fileName), "utf8"),
}));
const attributionBanner = `/*!\n${licenceFiles[1].source.trim()}\nLicence: https://polyformproject.org/licenses/noncommercial/1.0.0\n*/`;

export default defineConfig(({ mode }) => ({
  base: mode === "demo" ? "/Deep-Research-Studio/" : "/",
  define: {
    "import.meta.env.VITE_DEEPRESEARCH_API_ORIGIN": JSON.stringify(""),
  },
  plugins: [
    react(),
    {
      name: "distribute-licence-notices",
      apply: "build",
      generateBundle() {
        for (const asset of licenceFiles) {
          this.emitFile({ type: "asset", ...asset });
        }
      },
    },
  ],
  build: {
    rollupOptions: { output: { banner: attributionBanner } },
  },
  resolve: {
    alias: {
      "@/lib/api": path.resolve(__dirname, mode === "demo" ? "src/lib/demoApi.ts" : "src/lib/api.ts"),
      "@": path.resolve(__dirname, "src"),
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: true },
    },
  },
  preview: {
    host: "127.0.0.1",
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: true },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
    // Playwright specs under e2e/ run via `playwright test`, not vitest.
    exclude: ["**/node_modules/**", "**/dist/**", "e2e/**"],
  },
}));
