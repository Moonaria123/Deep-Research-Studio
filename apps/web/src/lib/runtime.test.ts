import { afterEach, describe, expect, it, vi } from "vitest";

describe("demo runtime", () => {
  afterEach(() => vi.resetModules());

  it("uses process-memory storage in demo mode", async () => {
    vi.stubEnv("MODE", "demo");
    const { installDemoRuntime } = await import("./runtime");
    installDemoRuntime();
    localStorage.setItem("preview", "yes");
    expect(localStorage.getItem("preview")).toBe("yes");
    expect(document.documentElement.dataset.demo).toBe("true");
    vi.unstubAllEnvs();
  });
});

describe("route strategy", () => {
  it("uses reload-safe hash routing only for packaged/static runtimes", async () => {
    const { usesHashRouting } = await import("./runtime");
    expect(usesHashRouting("desktop")).toBe(true);
    expect(usesHashRouting("demo")).toBe(true);
    expect(usesHashRouting("production")).toBe(false);
    expect(usesHashRouting("development")).toBe(false);
  });
});
