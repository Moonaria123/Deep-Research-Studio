import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { WikiInspector, findWikiTopicForNode } from "./WikiInspector";
import type { KnowledgeGraphNode, KnowledgeGraphWikiMap, KnowledgeGraphWikiTopic } from "@/lib/types";
import { summarizeWikiPartial } from "@/lib/wikiPartialSummary";

function topic(id: string, title: string, sessionId: string, attrs: Record<string, unknown> = {}): KnowledgeGraphWikiTopic {
  return {
    id,
    snapshot_id: "snapshot-1",
    key: `wiki:${id}`,
    title,
    summary: `${title} summary`,
    depth: 2,
    parent_id: "",
    session_ids: [sessionId],
    source_message_ids: [`msg-${sessionId}`],
    attrs: {
      kind: "topic",
      session_id: sessionId,
      session_title: `Session ${sessionId}`,
      concept_key: title.toLowerCase().replaceAll(" ", "-"),
      confidence: "high",
      quality: "grounded",
      evidence_refs: [{ type: "message", id: `msg-${sessionId}`, excerpt: `${title} evidence.` }],
      ...attrs,
    },
  };
}

function wikiMap(status: "succeeded" | "partial" = "succeeded"): KnowledgeGraphWikiMap {
  const main = topic("topic-main", "Risk taxonomy", "sess-1");
  const peer = topic("topic-peer", "Climate policy", "sess-2");
  const errors = [
    {
      session_id: "sess-failed",
      error_code: "INVALID_EVIDENCE_REF",
      validation_error: "INVALID_EVIDENCE_REF",
      structured_strategy: "json_schema_strict",
      request_id: "req-hidden",
      raw_transcript: "RAW_TRANSCRIPT_SHOULD_NOT_RENDER",
    },
  ];
  const stats = {
    topics: 2,
    tree_edges: 0,
    cross_links: 1,
    attempted_sessions: 2,
    succeeded_sessions: 1,
    failed_sessions: 1,
    skipped_sessions: 0,
    evidence_ref_dropped: 1,
    errors,
  };
  return {
    workspace_id: "default",
    layout_mode: "wiki_map",
    active_graph_version: { id: "kgv-2", workspace_id: "default", status: "active", is_active: true },
    active_snapshot: {
      id: "snapshot-1",
      workspace_id: "default",
      graph_version_id: "kgv-1",
      status,
      is_active: true,
      stale: false,
      stale_reason: "",
      finished_at: 1_750_000_000,
      stats,
    },
    stale: false,
    stale_reason: "",
    stats,
    topics: [main, peer],
    tree_edges: [],
    cross_links: [
      {
        id: "cross-1",
        snapshot_id: "snapshot-1",
        source: main.id,
        target: peer.id,
        type: "shared_kg_evidence",
        evidence_type: "shared_kg_evidence",
        evidence_refs: [{ type: "message", id: "msg-sess-1" }],
        attrs: { reason: "Both sessions share message evidence.", confidence: "high" },
      },
    ],
    last_refresh_job: {
      id: "job-1",
      snapshot_id: "snapshot-1",
      graph_version_id: "kgv-1",
      workspace_id: "default",
      backend: "thread",
      backend_job_id: "job-1",
      status,
      stage: status,
      progress: 1,
      attempted_sessions: 2,
      succeeded_sessions: 1,
      skipped_sessions: 0,
      failed_sessions: status === "partial" ? 1 : 0,
      stats,
    },
  };
}

function nodeFor(topic: KnowledgeGraphWikiTopic): KnowledgeGraphNode {
  return {
    id: topic.id,
    type: "wikiTopic",
    key: topic.key,
    title: topic.title,
    summary: topic.summary,
    confidence: 1,
    visible: true,
    attrs: topic.attrs,
  };
}

describe("WikiInspector", () => {
  it("finds the matching Wiki topic for a graph node", () => {
    const map = wikiMap();
    expect(findWikiTopicForNode(nodeFor(map.topics[0]), map)?.id).toBe("topic-main");
  });

  it("renders freshness, partial diagnostics, evidence, and secret-free errors", () => {
    const map = wikiMap("partial");
    render(
      <WikiInspector
        selectedNode={nodeFor(map.topics[0])}
        selectedTopic={map.topics[0]}
        wikiMap={map}
        wikiPartial={summarizeWikiPartial(map)}
        visibleTopicIds={new Set(["topic-main", "topic-peer"])}
        onSelectTopic={vi.fn()}
        onOpenSession={vi.fn()}
      />,
    );

    expect(screen.getByLabelText("Wiki Inspector")).toHaveTextContent("Risk taxonomy");
    expect(screen.getByText("Concept key:")).toBeInTheDocument();
    expect(screen.getByText("Wiki snapshot is from an older graph version.")).toBeInTheDocument();
    expect(screen.getByText("Wiki snapshot is partial.")).toBeInTheDocument();
    expect(screen.getByText("Evidence 引用不合法")).toBeInTheDocument();
    expect(screen.getByText("这是旧快照常见错误；请在当前版本重新 Refresh Wiki，让系统用新的 evidence allowlist 生成。")).toBeInTheDocument();
    expect(screen.getByText("1 dropped refs")).toBeInTheDocument();
    expect(screen.getByText("Risk taxonomy evidence.")).toBeInTheDocument();
    expect(screen.queryByText("req-hidden")).not.toBeInTheDocument();
    expect(screen.queryByText("RAW_TRANSCRIPT_SHOULD_NOT_RENDER")).not.toBeInTheDocument();
  });

  it("selects visible peer topics and explains hidden peers without mutating filters", () => {
    const map = wikiMap();
    const onSelectTopic = vi.fn();
    render(
      <WikiInspector
        selectedNode={nodeFor(map.topics[0])}
        selectedTopic={map.topics[0]}
        wikiMap={map}
        wikiPartial={null}
        visibleTopicIds={new Set(["topic-main"])}
        onSelectTopic={onSelectTopic}
        onOpenSession={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Climate policy" }));

    expect(onSelectTopic).not.toHaveBeenCalled();
    expect(screen.getByText("Peer topic is hidden by the current view.")).toBeInTheDocument();
    expect(screen.getByText("Climate policy is not visible under the current Session, search, or type filters.")).toBeInTheDocument();
  });

  it("opens the source session from the selected topic", () => {
    const map = wikiMap();
    const onOpenSession = vi.fn();
    render(
      <WikiInspector
        selectedNode={nodeFor(map.topics[0])}
        selectedTopic={map.topics[0]}
        wikiMap={map}
        wikiPartial={null}
        visibleTopicIds={new Set(["topic-main", "topic-peer"])}
        onSelectTopic={vi.fn()}
        onOpenSession={onOpenSession}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Open source session" }));

    expect(onOpenSession).toHaveBeenCalledWith("sess-1");
  });

  it("shows the source session title above the source session action", () => {
    const map = wikiMap();
    render(
      <WikiInspector
        selectedNode={nodeFor(map.topics[0])}
        selectedTopic={map.topics[0]}
        wikiMap={map}
        wikiPartial={null}
        visibleTopicIds={new Set(["topic-main", "topic-peer"])}
        onSelectTopic={vi.fn()}
        onOpenSession={vi.fn()}
      />,
    );

    expect(screen.getByText("Source session")).toBeInTheDocument();
    expect(screen.getByText("Session sess-1")).toBeInTheDocument();
  });

  it("falls back to a short session id when the title is missing", () => {
    const map = wikiMap();
    const selectedTopic = {
      ...map.topics[0],
      session_ids: ["session-abcdef1234567890"],
      attrs: { ...map.topics[0].attrs, session_id: "session-abcdef1234567890", session_title: "" },
    };
    render(
      <WikiInspector
        selectedNode={nodeFor(selectedTopic)}
        selectedTopic={selectedTopic}
        wikiMap={{ ...map, topics: [selectedTopic, map.topics[1]] }}
        wikiPartial={null}
        visibleTopicIds={new Set(["topic-main", "topic-peer"])}
        onSelectTopic={vi.fn()}
        onOpenSession={vi.fn()}
      />,
    );

    expect(screen.getByText("session-...7890")).toBeInTheDocument();
  });

  it("hides the source session block when no session id is available", () => {
    const map = wikiMap();
    const selectedTopic = {
      ...map.topics[0],
      session_ids: [],
      attrs: { ...map.topics[0].attrs, session_id: "", session_title: "" },
    };
    render(
      <WikiInspector
        selectedNode={nodeFor(selectedTopic)}
        selectedTopic={selectedTopic}
        wikiMap={{ ...map, topics: [selectedTopic, map.topics[1]] }}
        wikiPartial={null}
        visibleTopicIds={new Set(["topic-main", "topic-peer"])}
        onSelectTopic={vi.fn()}
        onOpenSession={vi.fn()}
      />,
    );

    expect(screen.queryByText("Source session")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Open source session" })).not.toBeInTheDocument();
  });

  it("offers explicit branch focus controls without changing expansion state", () => {
    const map = wikiMap();
    const onFocusTopic = vi.fn();
    const onClearFocus = vi.fn();
    const { rerender } = render(
      <WikiInspector
        selectedNode={nodeFor(map.topics[0])}
        selectedTopic={map.topics[0]}
        wikiMap={map}
        wikiPartial={null}
        visibleTopicIds={new Set(["topic-main", "topic-peer"])}
        onSelectTopic={vi.fn()}
        onOpenSession={vi.fn()}
        onFocusTopic={onFocusTopic}
        onClearFocus={onClearFocus}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Focus branch" }));
    expect(onFocusTopic).toHaveBeenCalledWith("topic-main");

    rerender(
      <WikiInspector
        selectedNode={nodeFor(map.topics[0])}
        selectedTopic={map.topics[0]}
        wikiMap={map}
        wikiPartial={null}
        visibleTopicIds={new Set(["topic-main", "topic-peer"])}
        focusedTopicId="topic-main"
        onSelectTopic={vi.fn()}
        onOpenSession={vi.fn()}
        onFocusTopic={onFocusTopic}
        onClearFocus={onClearFocus}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Return to full map" }));
    expect(onClearFocus).toHaveBeenCalledOnce();
  });

  it("opens navigable evidence and leaves unknown references inert", () => {
    const map = wikiMap();
    const selectedTopic = {
      ...map.topics[0],
      attrs: {
        ...map.topics[0].attrs,
        evidence_refs: [
          { type: "source", id: "source:d1", excerpt: "Source evidence." },
          { type: "unknown", id: "opaque", excerpt: "Unknown evidence." },
        ],
      },
    };
    const onOpenEvidence = vi.fn();
    render(
      <WikiInspector
        selectedNode={nodeFor(selectedTopic)}
        selectedTopic={selectedTopic}
        wikiMap={{ ...map, topics: [selectedTopic, map.topics[1]] }}
        wikiPartial={null}
        visibleTopicIds={new Set(["topic-main", "topic-peer"])}
        onSelectTopic={vi.fn()}
        onOpenSession={vi.fn()}
        onOpenEvidence={onOpenEvidence}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /source source:d1/i }));
    expect(onOpenEvidence).toHaveBeenCalledWith("/library/sources/d1");
    expect(screen.queryByRole("button", { name: /unknown opaque/i })).not.toBeInTheDocument();
    expect(screen.getByText("(unavailable)")).toBeInTheDocument();
  });

  it("shows selected cross-link reason, endpoints, and evidence", () => {
    const map = wikiMap();
    render(
      <WikiInspector
        selectedNode={nodeFor(map.topics[0])}
        selectedTopic={map.topics[0]}
        selectedCrossLink={map.cross_links[0]}
        wikiMap={map}
        wikiPartial={null}
        visibleTopicIds={new Set(["topic-main", "topic-peer"])}
        onSelectTopic={vi.fn()}
        onOpenSession={vi.fn()}
      />,
    );

    const detail = screen.getByLabelText("Cross-link detail");
    expect(detail).toHaveTextContent("Risk taxonomy");
    expect(detail).toHaveTextContent("Climate policy");
    expect(detail).toHaveTextContent("Both sessions share message evidence.");
    expect(detail).toHaveTextContent("msg-sess-1");
  });
});
