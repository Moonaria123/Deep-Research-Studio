import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ConfidenceTag } from "./ConfidenceTag";

describe("ConfidenceTag", () => {
  it("renders the qualitative label for known confidence values", () => {
    render(<ConfidenceTag label="established" />);
    expect(screen.getByText("Established")).toBeInTheDocument();
  });

  it("frames contested as divergence, not failure", () => {
    render(<ConfidenceTag label="contested" />);
    const tag = screen.getByText("Contested");
    expect(tag).toBeInTheDocument();
    expect(screen.getByLabelText(/Confidence: Contested/)).toBeInTheDocument();
  });

  it("renders nothing for unknown or empty labels", () => {
    const { container } = render(<ConfidenceTag label="" />);
    expect(container).toBeEmptyDOMElement();
    const { container: c2 } = render(<ConfidenceTag label="bogus" />);
    expect(c2).toBeEmptyDOMElement();
  });
});
