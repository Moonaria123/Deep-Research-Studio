import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { WikiSessionRootNode } from "./WikiSessionRootNode";
import type { KnowledgeGraphNode } from "@/lib/types";

vi.mock("@/components/graph/GraphNodeHandles", () => ({ GraphNodeHandles: () => null }));

function rootNode(attrs: Record<string, unknown>): KnowledgeGraphNode {
  return {
    id: "root-1",
    type: "wikiTopic",
    key: "wiki:session:sess-1",
    title: "Session 1",
    summary: "A bounded Session summary.",
    confidence: 1,
    visible: true,
    attrs: { kind: "session_root", session_id: "sess-1", ...attrs },
  };
}

function renderRoot(attrs: Record<string, unknown>) {
  render(
    <WikiSessionRootNode
      {...({
        id: "root-1",
        type: "wikiSessionRoot",
        selected: false,
        data: { graphNode: rootNode(attrs), label: "Session 1", selected: false, highlighted: false },
      } as any)}
    />,
  );
}

describe("WikiSessionRootNode", () => {
  it("renders topic count, refresh time, and carried/stale state", () => {
    renderRoot({
      topic_count: 3,
      last_refreshed_at: 1_750_000_000,
      generation_state: "carried",
      stale: true,
    });

    expect(screen.getByText("3 topics")).toBeInTheDocument();
    expect(screen.getByText(/Carried/)).toBeInTheDocument();
    expect(screen.getByText(/Stale/)).toBeInTheDocument();
    expect(screen.getByText(/Updated/)).toBeInTheDocument();
  });

  it("keeps old DTOs unknown instead of labeling them fresh", () => {
    renderRoot({});

    expect(screen.getByText("Topic count unknown")).toBeInTheDocument();
    expect(screen.queryByText(/Fresh/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Updated/)).not.toBeInTheDocument();
  });
});
