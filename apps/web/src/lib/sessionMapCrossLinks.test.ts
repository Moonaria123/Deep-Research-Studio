import { describe, expect, it } from "vitest";
import type { KnowledgeGraphNode, KnowledgeGraphSessionMap } from "@/lib/types";
import {
  buildSessionMapCrossLinkEdges,
  sessionMapCrossLinkedSessionIds,
  type SessionMapCrossLinkAnchor,
  type SessionMapSharedAssetAnchor,
} from "./sessionMapCrossLinks";

function memoryNode(key: string, sessionIds: string[]): KnowledgeGraphNode {
  return {
    id: `node-${key}`,
    type: "memory",
    key,
    title: "Shared memory",
    summary: "Shared memory summary",
    confidence: 1,
    visible: true,
    attrs: {},
    referencing_sessions: sessionIds.map((sessionId) => ({
      node_key: `session:${sessionId}`,
      session_id: sessionId,
      title: `Session ${sessionId}`,
      summary: "",
    })),
  };
}

function anchor(memory: KnowledgeGraphNode, fromSessionId: string): SessionMapCrossLinkAnchor {
  return {
    memoryKey: memory.key,
    fromSessionId,
    memoryFlowId: `memory:session:${fromSessionId}:${memory.key}`,
    memoryNode: memory,
  };
}

function assetAnchor(
  assetKey: string,
  fromSessionId: string,
  assetKind: "source" | "note" | "report" | "memory",
): SessionMapSharedAssetAnchor {
  return {
    assetKey,
    assetKind,
    fromSessionId,
    assetFlowId: `${assetKind}:session:${fromSessionId}:${assetKey}`,
  };
}

function sessionLane(sessionId: string, sourceKeys: string[], memoryKeys: string[]): KnowledgeGraphSessionMap["sessions"][number] {
  return {
    session: {
      id: `node-${sessionId}`,
      type: "session",
      key: `session:${sessionId}`,
      title: `Session ${sessionId}`,
      summary: "",
      confidence: 1,
      visible: true,
      attrs: { session_id: sessionId },
    },
    reports: [],
    notes: [],
    source_set: { count: sourceKeys.length, node_keys: sourceKeys, members: [] },
    memory_set: { count: memoryKeys.length, node_keys: memoryKeys, members: [] },
    counts: { reports: 0, notes: 0, sources: sourceKeys.length, memories: memoryKeys.length },
  };
}

describe("buildSessionMapCrossLinkEdges", () => {
  it("builds directed cross-session edges for visible expanded memory anchors", () => {
    const memory = memoryNode("memory:shared", ["sess-a", "sess-b"]);
    const edges = buildSessionMapCrossLinkEdges({
      memoryAnchors: [anchor(memory, "sess-a"), anchor(memory, "sess-b")],
      nodesByKey: { [memory.key]: memory },
      sessionFlowIdBySessionId: new Map([
        ["sess-a", "sessionCard:session:sess-a"],
        ["sess-b", "sessionCard:session:sess-b"],
      ]),
    });

    expect(edges.map((edge) => edge.id)).toEqual([
      "session-cross-memory:shared-sess-a-sess-b",
      "session-cross-memory:shared-sess-b-sess-a",
    ]);
    expect(edges[0]).toMatchObject({
      source: "memory:session:sess-a:memory:shared",
      target: "sessionCard:session:sess-b",
      fromSessionId: "sess-a",
      toSessionId: "sess-b",
    });
  });

  it("skips self references, hidden target sessions, and duplicate edge ids", () => {
    const memory = memoryNode("memory:shared", ["sess-a", "sess-b", "sess-c"]);
    const edges = buildSessionMapCrossLinkEdges({
      memoryAnchors: [anchor(memory, "sess-a"), anchor(memory, "sess-a")],
      nodesByKey: { [memory.key]: memory },
      sessionFlowIdBySessionId: new Map([
        ["sess-a", "sessionCard:session:sess-a"],
        ["sess-b", "sessionCard:session:sess-b"],
      ]),
    });

    expect(edges.map((edge) => edge.id)).toEqual(["session-cross-memory:shared-sess-a-sess-b"]);
  });

  it("falls back to nodesByKey referencing sessions when the anchor node is sparse", () => {
    const sparseMemory = { ...memoryNode("memory:shared", []), referencing_sessions: undefined };
    const fullMemory = memoryNode("memory:shared", ["sess-a", "sess-b"]);
    const edges = buildSessionMapCrossLinkEdges({
      memoryAnchors: [anchor(sparseMemory, "sess-a")],
      nodesByKey: { [fullMemory.key]: fullMemory },
      sessionFlowIdBySessionId: new Map([
        ["sess-a", "sessionCard:session:sess-a"],
        ["sess-b", "sessionCard:session:sess-b"],
      ]),
    });

    expect(edges).toHaveLength(1);
    expect(edges[0].target).toBe("sessionCard:session:sess-b");
  });

  it("builds shared source, note, report, and memory edges from visible anchors", () => {
    const memory = memoryNode("memory:shared", []);
    const edges = buildSessionMapCrossLinkEdges({
      memoryAnchors: [],
      sharedAnchors: [
        assetAnchor("source:shared", "sess-a", "source"),
        assetAnchor("source:shared", "sess-b", "source"),
        assetAnchor("note:shared", "sess-a", "note"),
        assetAnchor("note:shared", "sess-b", "note"),
        assetAnchor("report:shared", "sess-a", "report"),
        assetAnchor("report:shared", "sess-b", "report"),
        assetAnchor(memory.key, "sess-a", "memory"),
        assetAnchor(memory.key, "sess-b", "memory"),
      ],
      nodesByKey: { [memory.key]: memory },
      sessionFlowIdBySessionId: new Map([
        ["sess-a", "sessionCard:session:sess-a"],
        ["sess-b", "sessionCard:session:sess-b"],
      ]),
    });

    expect(edges.map((edge) => edge.id)).toEqual([
      "session-cross-source:shared-sess-a-sess-b",
      "session-cross-source:shared-sess-b-sess-a",
      "session-cross-note:shared-sess-a-sess-b",
      "session-cross-note:shared-sess-b-sess-a",
      "session-cross-report:shared-sess-a-sess-b",
      "session-cross-report:shared-sess-b-sess-a",
      "session-cross-memory:shared-sess-a-sess-b",
      "session-cross-memory:shared-sess-b-sess-a",
    ]);
    expect(edges[0]).toMatchObject({
      label: "cross-session shared source",
      source: "source:session:sess-a:source:shared",
      target: "sessionCard:session:sess-b",
    });
  });
});

describe("sessionMapCrossLinkedSessionIds", () => {
  it("returns sessions linked by shared sources, shared memories, and memory references", () => {
    const referencedMemory = memoryNode("memory:referenced", ["sess-c", "sess-d"]);
    const sessionMap = {
      workspace_id: "default",
      active_version: null,
      layout_mode: "session_tree",
      session_limit: 50,
      member_preview: 100,
      sessions: [
        sessionLane("sess-a", ["source:shared"], ["memory:one"]),
        sessionLane("sess-b", ["source:shared"], ["memory:shared"]),
        sessionLane("sess-c", ["source:solo"], ["memory:shared", referencedMemory.key]),
        sessionLane("sess-d", ["source:other"], []),
      ],
      nodes_by_key: {
        [referencedMemory.key]: referencedMemory,
      },
      edges: [],
    } satisfies KnowledgeGraphSessionMap;

    expect([...sessionMapCrossLinkedSessionIds(sessionMap)].sort()).toEqual(["sess-a", "sess-b", "sess-c", "sess-d"]);
  });

  it("marks sessions linked by the same stable note or report", () => {
    const note = { ...memoryNode("note:shared", []), type: "note", title: "Shared note" };
    const report = { ...memoryNode("report:shared", []), type: "report", title: "Shared report" };
    const first = sessionLane("sess-a", [], []);
    const second = sessionLane("sess-b", [], []);
    const third = sessionLane("sess-c", [], []);
    first.notes = [note];
    second.notes = [note];
    second.reports = [report];
    third.reports = [report];
    const sessionMap = {
      workspace_id: "default",
      active_version: null,
      layout_mode: "session_tree",
      session_limit: 50,
      member_preview: 100,
      sessions: [first, second, third],
      nodes_by_key: { [note.key]: note, [report.key]: report },
      edges: [],
    } satisfies KnowledgeGraphSessionMap;

    expect([...sessionMapCrossLinkedSessionIds(sessionMap)].sort()).toEqual(["sess-a", "sess-b", "sess-c"]);
  });
});
