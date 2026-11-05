import { describe, expect, it, vi, beforeAll } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { SourcesRail } from "./SourcesRail";
import type { EvidenceSource } from "@/lib/types";

const src = (id: string): EvidenceSource => ({
  chunk_id: 1,
  document_id: id,
  url: `https://e.com/${id}`,
  title: `Source ${id}`,
  text: "snippet",
  score: 0.9,
});

beforeAll(() => {
  // jsdom lacks scrollIntoView; SourceCard calls it when becoming active.
  Element.prototype.scrollIntoView = vi.fn();
});

describe("SourcesRail", () => {
  const sources = [src("a"), src("b"), src("c")];

  it("highlights the card whose index is in highlightIndexes", () => {
    const { container } = render(<SourcesRail sources={sources} highlightIndexes={[2]} />);
    const active = container.querySelectorAll('[data-source-active="true"]');
    expect(active.length).toBe(1);
    expect(active[0].id).toBe("source-2");
  });

  it("fires onSourcePin with the 1-based index on click", () => {
    const onSourcePin = vi.fn();
    const { container } = render(<SourcesRail sources={sources} onSourcePin={onSourcePin} />);
    fireEvent.click(container.querySelector("#source-3")!);
    expect(onSourcePin).toHaveBeenCalledWith(3);
  });

  it("renders an empty state without crashing", () => {
    const { getByText } = render(<SourcesRail sources={[]} />);
    expect(getByText(/No sources for this run/)).toBeInTheDocument();
  });

  it("renders local and archived-memory provenance with internal deep links", () => {
    const internal: EvidenceSource[] = [
      {
        chunk_id: -1,
        document_id: "note:n1",
        url: null,
        title: "Decision note",
        text: "Internal decision",
        score: 0.8,
        origin: "local",
        artifact_type: "library_note",
        artifact_id: "n1",
        source_status: "active",
      },
      {
        chunk_id: -2,
        document_id: "memory:m1",
        url: null,
        title: "Archived memory",
        text: "Prior result",
        score: 0.7,
        origin: "memory",
        artifact_type: "memory",
        artifact_id: "m1",
        source_status: "archived",
      },
    ];

    render(
      <MemoryRouter>
        <SourcesRail sources={internal} />
      </MemoryRouter>,
    );

    expect(screen.getByRole("link", { name: /Decision note/ })).toHaveAttribute("href", "/library/notes/n1");
    expect(screen.getByRole("link", { name: /Archived memory/ })).toHaveAttribute("href", "/memory/m1");
    expect(screen.getByText("origin: local")).toBeInTheDocument();
    expect(screen.getByText("origin: memory")).toBeInTheDocument();
    expect(screen.getByText("artifact: memory")).toBeInTheDocument();
    expect(screen.getByText("archived")).toBeInTheDocument();
  });
});
