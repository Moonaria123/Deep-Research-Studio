import { describe, expect, it } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { LimitationsPanel } from "./LimitationsPanel";
import type { Citation } from "@/lib/types";

const citations: Citation[] = [
  { id: "c1", document_id: "d1", url: "https://e.com/a", title: "Source A", snippet: "", score: 0.9 },
];

describe("LimitationsPanel", () => {
  it("renders nothing when there are no limitations", () => {
    const { container } = render(<LimitationsPanel text="" citations={citations} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the header and expands to reveal limitation text", () => {
    render(<LimitationsPanel text="Data is from 2023 [c1]." citations={citations} />);
    const toggle = screen.getByRole("button", { name: /limitations/i });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText(/Data is from 2023/)).toBeNull();
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(/Data is from 2023/)).toBeInTheDocument();
  });

  it("respects defaultOpen", () => {
    render(<LimitationsPanel text="Sample is small [c1]." citations={citations} defaultOpen />);
    expect(screen.getByText(/Sample is small/)).toBeInTheDocument();
  });
});
