import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { AnswerBody } from "./AnswerBody";
import type { Citation } from "@/lib/types";

const citations: Citation[] = [
  { id: "c1", document_id: "d1", url: "https://e.com/a", title: "Source A", snippet: "", score: 0.9 },
  { id: "c2", document_id: "d2", url: "https://e.com/b", title: "Source B", snippet: "", score: 0.7 },
];

describe("AnswerBody", () => {
  it("renders inline [cN] markers as anchored citation chips", () => {
    render(<AnswerBody text="Water boils at 100C [c1] at sea level [c2]." citations={citations} />);
    const chip1 = screen.getByRole("link", { name: "1" });
    const chip2 = screen.getByRole("link", { name: "2" });
    expect(chip1).toHaveAttribute("href", "#source-1");
    expect(chip1).toHaveAttribute("title", "Source A");
    expect(chip2).toHaveAttribute("href", "#source-2");
  });

  it("keeps plain text when there are no markers", () => {
    render(<AnswerBody text="No citations here." citations={[]} />);
    expect(screen.getByText("No citations here.")).toBeInTheDocument();
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("falls back to the marker id as title when the citation is missing", () => {
    render(<AnswerBody text="Unknown [c9] ref." citations={citations} />);
    expect(screen.getByRole("link", { name: "9" })).toHaveAttribute("title", "c9");
  });

  it("does not transform [cN] inside inline code spans", () => {
    render(<AnswerBody text="Use the `[c1]` token literally." citations={citations} />);
    // The marker stays verbatim inside the code span and renders no chip link.
    expect(screen.getByText("[c1]")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "1" })).toBeNull();
  });

  it("does not wrap sentences when claim_map is absent", () => {
    const { container } = render(<AnswerBody text="Water boils at 100C." citations={citations} />);
    expect(container.querySelector("[data-claim]")).toBeNull();
  });

  it("wraps a matched sentence and fires onClaimPin on click", () => {
    const onClaimPin = vi.fn();
    const { container } = render(
      <AnswerBody
        text="Solar is renewable [c1]. Wind is clean [c2]."
        citations={citations}
        claimMap={{ "Solar is renewable [c1].": ["c1"], "Wind is clean [c2].": ["c2"] }}
        onClaimPin={onClaimPin}
      />,
    );
    const spans = container.querySelectorAll("[data-claim]");
    expect(spans.length).toBe(2);
    const first = container.querySelector('[data-citation-ids="c1"]');
    expect(first).not.toBeNull();
    expect(first).toHaveTextContent("Solar is renewable");
    fireEvent.click(first!);
    expect(onClaimPin).toHaveBeenCalledWith(["c1"]);
  });

  it("marks a sentence active when activeIds intersects its citations", () => {
    const { container } = render(
      <AnswerBody
        text="Solar is renewable [c1]."
        citations={citations}
        claimMap={{ "Solar is renewable [c1].": ["c1"] }}
        activeIds={["c1"]}
      />,
    );
    expect(container.querySelector('[data-claim-active="true"]')).not.toBeNull();
  });

  it("leaves unmatched sentences unwrapped (graceful no-op)", () => {
    const { container } = render(
      <AnswerBody
        text="Totally unrelated prose."
        citations={citations}
        claimMap={{ "Some other claim.": ["c1"] }}
      />,
    );
    expect(container.querySelector("[data-claim]")).toBeNull();
    expect(screen.getByText(/Totally unrelated prose/)).toBeInTheDocument();
  });
});
