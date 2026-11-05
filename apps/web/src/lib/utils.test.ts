import { describe, expect, it } from "vitest";
import { cn, domainOf, formatTime } from "./utils";

describe("cn", () => {
  it("merges and dedupes conflicting tailwind classes", () => {
    expect(cn("px-2", "px-4")).toBe("px-4");
    expect(cn("text-sm", false && "hidden", "font-bold")).toBe("text-sm font-bold");
  });
});

describe("domainOf", () => {
  it("returns hostname without www", () => {
    expect(domainOf("https://www.example.com/a/b?q=1")).toBe("example.com");
    expect(domainOf("http://docs.python.org/3")).toBe("docs.python.org");
  });

  it("falls back to the raw value for non-URLs", () => {
    expect(domainOf("not a url")).toBe("not a url");
  });
});

describe("formatTime", () => {
  it("returns empty string for falsy input", () => {
    expect(formatTime(undefined)).toBe("");
    expect(formatTime(0)).toBe("");
  });

  it("formats an epoch-seconds value", () => {
    expect(formatTime(1_700_000_000)).not.toBe("");
  });
});
