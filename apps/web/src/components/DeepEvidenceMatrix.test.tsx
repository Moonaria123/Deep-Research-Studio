import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { DeepEvidenceMatrix } from "./DeepEvidenceMatrix";
import type { EvidenceMatrixRow } from "@/lib/types";

describe("DeepEvidenceMatrix", () => {
  const rows: EvidenceMatrixRow[] = [
    {
      subtask: "Market structure",
      queries: ["market query"],
      notes: "Concentrated suppliers",
      gaps: ["pricing"],
      source_count: 4,
      status: "researched",
    },
    {
      subtask: "Pricing gap",
      queries: ["pricing query"],
      notes: "Pricing still incomplete",
      gaps: [],
      source_count: 1,
      status: "gap_closed",
    },
  ];

  it("renders Deep evidence matrix rows with source counts and gaps", () => {
    render(<DeepEvidenceMatrix rows={rows} />);

    expect(screen.getByText("Evidence matrix")).toBeInTheDocument();
    expect(screen.getByText("Market structure")).toBeInTheDocument();
    expect(screen.getByText("Concentrated suppliers")).toBeInTheDocument();
    expect(screen.getByText("4")).toBeInTheDocument();
    expect(screen.getByText("pricing")).toBeInTheDocument();
    expect(screen.getByText("gap closed")).toBeInTheDocument();
  });

  it("renders nothing for an empty matrix", () => {
    const { container } = render(<DeepEvidenceMatrix rows={[]} />);
    expect(container.firstChild).toBeNull();
  });
});
