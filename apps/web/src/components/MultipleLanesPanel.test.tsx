import { describe, expect, it } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MultipleLanesPanel } from "./MultipleLanesPanel";
import type { EvidenceSource } from "@/lib/types";

const source = (id: string, title: string): EvidenceSource => ({
  chunk_id: 1,
  document_id: id,
  url: `https://e.com/${id}`,
  title,
  text: "snippet",
  score: 0.9,
});

const lanes = [
  { query: "history of solar cells", sources: [source("d1", "Solar A"), source("d2", "Solar B")] },
  { query: "current efficiency records", sources: [source("d3", "Records")] },
];

describe("MultipleLanesPanel", () => {
  it("renders nothing without lanes", () => {
    const { container } = render(<MultipleLanesPanel lanes={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("lists each sub-query with its source count and hides sources until expanded", () => {
    render(<MultipleLanesPanel lanes={lanes} />);
    expect(screen.getByText("history of solar cells")).toBeInTheDocument();
    expect(screen.getByText("current efficiency records")).toBeInTheDocument();
    expect(screen.getByText("2 sources")).toBeInTheDocument();
    expect(screen.getByText("1 source")).toBeInTheDocument();

    // Lane sources are collapsed initially.
    expect(screen.queryByText("Solar A")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /history of solar cells/i }));
    expect(screen.getByText("Solar A")).toBeInTheDocument();
    expect(screen.getByText("Solar B")).toBeInTheDocument();
    // The other lane stays collapsed.
    expect(screen.queryByText("Records")).toBeNull();
  });

  it("shows only sources adopted by the final evidence snapshot", () => {
    render(<MultipleLanesPanel lanes={lanes} adoptedSources={[source("d3", "Records")]} />);

    expect(screen.getByText("0 sources")).toBeInTheDocument();
    expect(screen.getByText("1 source")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /history of solar cells/i }));
    expect(screen.getByText("No sources retrieved for this lane.")).toBeInTheDocument();
    expect(screen.queryByText("Solar A")).toBeNull();
  });
});
