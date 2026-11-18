import { describe, expect, it } from "vitest";

import { resolveApiBase } from "./apiOrigin";

describe("resolveApiBase", () => {
  it("keeps ordinary web builds on the relative API path", () => {
    expect(resolveApiBase(undefined)).toBe("/api/v1");
    expect(resolveApiBase("")).toBe("/api/v1");
  });

  it("targets the fixed loopback API when given the desktop origin", () => {
    expect(resolveApiBase("http://127.0.0.1:18789/ ")).toBe("http://127.0.0.1:18789/api/v1");
  });
});
