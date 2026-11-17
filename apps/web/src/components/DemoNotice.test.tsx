import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DemoNotice } from "./DemoNotice";

const runtimeState = vi.hoisted(() => ({ isDemo: false }));

vi.mock("@/lib/runtime", () => ({
  get IS_DEMO() { return runtimeState.isDemo; },
}));

describe("DemoNotice", () => {
  beforeEach(() => {
    runtimeState.isDemo = false;
  });

  it("stays hidden outside the public demo build", () => {
    render(<DemoNotice />);

    expect(screen.queryByRole("note", { name: "Demo preview notice" })).toBeNull();
  });

  it("links every demo route back to the public GitHub repository", () => {
    runtimeState.isDemo = true;
    render(<DemoNotice />);

    const link = screen.getByRole("link", { name: "View Deep Research Studio on GitHub" });
    expect(link).toHaveAttribute("href", "https://github.com/Moonaria123/Deep-Research-Studio");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noreferrer");
  });
});
