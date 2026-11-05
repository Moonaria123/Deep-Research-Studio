import { describe, expect, it } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ResearchProcessPanel } from "./ResearchProcessPanel";
import type { GroundedAnswer } from "@/lib/types";

const section = (title: string, text: string): { title: string; answer: GroundedAnswer } => ({
  title,
  answer: { text, citations: [], claim_map: {}, coverage: 1 },
});

describe("ResearchProcessPanel", () => {
  it("renders nothing without sections", () => {
    const { container } = render(<ResearchProcessPanel sections={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("is collapsed by default and hides the section drafts until expanded", () => {
    render(<ResearchProcessPanel sections={[section("Background", "Draft detail here.")]} />);
    const toggle = screen.getByRole("button", { name: /research process/i });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText(/Draft detail here/)).toBeNull();

    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("Background")).toBeInTheDocument();
  });
});
