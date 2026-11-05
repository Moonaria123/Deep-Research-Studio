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
