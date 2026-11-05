import { expect, test } from "@playwright/test";

const routes = [
  ["", "Welcome to Deep Research Studio"],
  ["history", "History"],
  ["history/archived", "Archived"],
  ["library", "Library"],
  ["library/reports", "Reports"],
  ["library/notes", "Notes"],
  ["memory", "Memory"],
  ["graph", "Knowledge Graph"],
  ["settings", "Settings"],
] as const;
const previewUrl = (route = "") => `http://127.0.0.1:4174/Deep-Research-Studio/#/${route}`;
const browserIssues = new WeakMap<object, string[]>();

test.beforeEach(async ({ page }) => {
  const issues: string[] = [];
  browserIssues.set(page, issues);
  page.on("pageerror", (error) => issues.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") issues.push(`console: ${message.text()}`);
  });
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.origin !== "http://127.0.0.1:4174") {
      issues.push(`external request: ${url.href}`);
    }
    if (url.pathname.startsWith("/api/")) issues.push(`API request: ${url.pathname}`);
  });
});

test.afterEach(async ({ page }) => {
  expect(browserIssues.get(page) ?? [], "Demo browser errors or live requests").toEqual([]);
});

test("all primary preview routes render from preset data", async ({ page }) => {
  for (const [route, heading] of routes) {
    await page.goto(previewUrl(route));
    await expect(page.getByLabel("Demo preview notice")).toBeVisible();
    await expect(page.getByText(heading, { exact: false }).first()).toBeVisible();
  }
});

test("four modes open real sanitized completed reports", async ({ page }) => {
  for (const mode of ["fast", "multiple", "deep", "council"] as const) {
    await page.goto(previewUrl(`run/demo-${mode}`));
    await expect(page.getByText("SQLite", { exact: false }).first()).toBeVisible();
    await expect(page.getByText(/sources cited/i)).toBeVisible();
  }
});

test("visitor state is memory-only and resets on reload", async ({ page }) => {
  await page.goto(previewUrl("settings"));
  await page.evaluate(() => localStorage.setItem("demo-probe", "transient"));
  expect(await page.evaluate(() => localStorage.getItem("demo-probe"))).toBe("transient");
  await page.reload();
  expect(await page.evaluate(() => localStorage.getItem("demo-probe"))).toBeNull();
  await expect(page.locator('input[type="password"]').first()).toBeDisabled({ timeout: 2_000 }).catch(async () => {
    await expect(page.locator('input[type="password"]').first()).toHaveCSS("pointer-events", "none");
  });
});

test("preset run replays without model interaction", async ({ page }) => {
  await page.goto(previewUrl());
  await page.getByRole("button", { name: /Which search query operators/i }).click();
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page).toHaveURL(/#\/run\/demo-fast$/);
  await expect(page.getByText(/MATCH operator/i).first()).toBeVisible();
  await expect(page.getByLabel("正在核验报告")).toBeVisible();
  await expect(page.getByLabel("核验完成")).toBeVisible({ timeout: 5_000 });
});

test("low confidence warning can be acknowledged or replayed without persistence", async ({ page }) => {
  await page.goto(previewUrl("run/demo-fast"));
  await page.getByRole("button", { name: "查看低置信度警告" }).click();
  await expect(page.getByRole("dialog", { name: "低置信度警告" })).toBeVisible();
  await page.getByRole("button", { name: "确认并忽略" }).click();
  await expect(page.getByRole("button", { name: "低置信度风险已确认" })).toBeVisible();

  await page.reload();
  await page.getByRole("button", { name: "查看低置信度警告" }).click();
  await page.getByRole("button", { name: "重新出具" }).click();
  await expect(page).toHaveURL(/#\/run\/demo-regenerated$/);
  await expect(page.getByLabel("正在核验报告")).toBeVisible();
});

test("dangerous actions are disabled while preview navigation remains available", async ({ page }) => {
  await page.goto(previewUrl("library"));
  await expect(page.getByRole("button", { name: "Index" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Import" })).toBeDisabled();
  await page.goto(previewUrl("library/notes"));
  await expect(page.getByRole("button", { name: "Add note" })).toBeDisabled();
  await page.goto(previewUrl("library/reports"));
  await expect(page.getByRole("button", { name: "Upload" })).toBeDisabled();

  await page.goto(previewUrl("memory"));
  await expect(page.getByRole("button", { name: "Refresh", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Add memory" })).toBeDisabled();
  await expect(page.getByRole("button", { name: /(?:Pin|Unpin) memory/ }).first()).toBeDisabled();

  await page.goto(previewUrl("graph"));
  await expect(page.getByRole("button", { name: "Rebuild graph" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Refresh Wiki" })).toBeDisabled();
  await expect(page.getByText("FTS5 indexing and ranking", { exact: false }).first()).toBeVisible();
  await page.getByRole("button", { name: "Wiki Map" }).click();
  await expect(page.getByText("FTS5 query language", { exact: false }).first()).toBeVisible();
});

test("settings dialogs are inspectable but every key and mutation control is disabled", async ({ page }) => {
  await page.goto(previewUrl("settings"));
  const passwordInputs = page.locator('input[type="password"]');
  await expect(passwordInputs).toHaveCount(4);
  for (let index = 0; index < 4; index += 1) await expect(passwordInputs.nth(index)).toBeDisabled();
  await expect(page.getByRole("button", { name: "Test brave" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Save search providers" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Clear retrieval cache" })).toBeDisabled();

  await page.getByRole("button", { name: "Add provider" }).click();
  const providerDialog = page.getByRole("dialog", { name: "Add provider" });
  await expect(providerDialog).toBeVisible();
  await expect(providerDialog.getByLabel("API key")).toBeDisabled();
  await expect(providerDialog.getByRole("button", { name: "Add provider" })).toBeDisabled();
  await page.keyboard.press("Escape");

  await page.getByRole("button", { name: "Manage models demo-preview" }).click();
  const modelDialog = page.getByRole("dialog", { name: "Model management" });
  await expect(modelDialog).toBeVisible();
  await expect(modelDialog.getByRole("button", { name: "Fetch available models" })).toBeDisabled();
  await expect(modelDialog.getByLabel("Add model id")).toBeDisabled();
  await expect(modelDialog.getByRole("button", { name: "Add", exact: true })).toBeDisabled();
  await expect(modelDialog.getByText(/model discovery, editing, testing, and deletion are disabled/i)).toBeVisible();
});

test("session defaults and report memory saves are disabled", async ({ page }) => {
  await page.goto(previewUrl("run/demo-fast"));
  await expect(page.getByRole("button", { name: "Save to memory" })).toBeDisabled();
  await page.goto(previewUrl("session/session-fast"));
  await expect(page.getByRole("button", { name: /session default/i })).toBeDisabled();
  await page.getByRole("button", { name: "Options" }).click();
  await page.getByRole("button", { name: "Show" }).click();
  await expect(page.getByRole("button", { name: "Customize this session" })).toBeDisabled();
});
