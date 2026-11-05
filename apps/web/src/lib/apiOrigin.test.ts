import { describe, expect, it } from "vitest";

import { resolveApiBase } from "./apiOrigin";

describe("resolveApiBase", () => {
  it("keeps ordinary web builds on the relative API path", () => {
    expect(resolveApiBase(undefined)).toBe("/api/v1");
    expect(resolveApiBase("")).toBe("/api/v1");
  });
});
