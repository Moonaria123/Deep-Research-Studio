import { describe, expect, it } from "vitest";
import type { KnowledgeGraphSessionLane, KnowledgeGraphWikiMap, KnowledgeGraphWikiTopic } from "@/lib/types";
import {
  defaultSelectedSessionIds,
  extractSessionIdFromLane,
  filterSessionLanes,
  filterWikiEdges,
  filterWikiTopicsWithCrossLinkClosure,
  highConfidenceCrossLinkedSessionIds,
  resolveSessionFilterSelection,
  sessionFilterOptions,
} from "./graphSessionFilter";

function lane(id: string, title = id): KnowledgeGraphSessionLane {
  return {
    session: {
      id: `node-${id}`,
      type: "session",
      key: `session:${id}`,
      title,
      summary: "",
      attrs: { session_id: id },
      confidence: 1,
      visible: true,
    },
    reports: [],
    notes: [],
    source_set: { count: 0, node_keys: [], members: [] },
    memory_set: { count: 0, node_keys: [], members: [] },
    counts: { reports: 0, notes: 0, sources: 0, memories: 0 },
  };
}

function topic(id: string, sessionIds: string[]): KnowledgeGraphWikiTopic {
  return {
    id,
    snapshot_id: "snap-1",
    key: `wiki:${id}`,
    title: id,
    summary: "",
    depth: 1,
    parent_id: "",
    session_ids: sessionIds,
    source_message_ids: [],
    attrs: {},
  };
}

function wikiMap(): KnowledgeGraphWikiMap {
  return {
    workspace_id: "default",
    layout_mode: "wiki_map",
    active_graph_version: null,
    active_snapshot: null,
    stale: false,
    stale_reason: "",
    stats: {},
    topics: [topic("a", ["sess-1"]), topic("b", ["sess-2"]), topic("c", ["sess-3"])],
    tree_edges: [
      { id: "tree-ab", snapshot_id: "snap-1", source: "a", target: "b", type: "parent_child", attrs: {} },
      { id: "tree-bc", snapshot_id: "snap-1", source: "b", target: "c", type: "parent_child", attrs: {} },
    ],
    cross_links: [
      {
        id: "cross-ac",
        snapshot_id: "snap-1",
        source: "a",
        target: "c",
        type: "shared_kg_evidence",
        evidence_type: "shared_kg_evidence",
        evidence_refs: [],
        attrs: { confidence: "high" },
      },
    ],
    last_refresh_job: null,
  };
}

describe("graphSessionFilter", () => {
  it("extracts stable session ids and display options from lanes", () => {
    const withKeyFallback = lane("sess-1", "First");
    withKeyFallback.session.attrs = {};

    expect(extractSessionIdFromLane(withKeyFallback)).toBe("sess-1");
    expect(sessionFilterOptions([withKeyFallback])).toEqual([{ id: "sess-1", title: "First" }]);
  });

  it("selects the first eight lanes as the default recent set", () => {
    const lanes = Array.from({ length: 10 }, (_, index) => lane(`sess-${index + 1}`));

    expect([...defaultSelectedSessionIds(lanes)]).toEqual([
      "sess-1",
      "sess-2",
      "sess-3",
      "sess-4",
      "sess-5",
      "sess-6",
      "sess-7",
      "sess-8",
    ]);
  });

  it("resolves all and manual selections without auto-resetting empty manual state", () => {
    const lanes = [lane("sess-1"), lane("sess-2")];

    expect([...resolveSessionFilterSelection({ mode: "all", lanes, manualSelectedIds: new Set() })]).toEqual([
      "sess-1",
      "sess-2",
    ]);
    expect([...resolveSessionFilterSelection({ mode: "manual", lanes, manualSelectedIds: new Set() })]).toEqual([]);
    expect([
      ...resolveSessionFilterSelection({
        mode: "manual",
        lanes,
        manualSelectedIds: new Set(["sess-2", "missing"]),
      }),
    ]).toEqual(["sess-2"]);
  });

  it("filters Session Map lanes by effective selection", () => {
    const lanes = [lane("sess-1"), lane("sess-2")];

    expect(filterSessionLanes(lanes, new Set(["sess-2"])).map((item) => item.session.title)).toEqual(["sess-2"]);
  });

  it("filters Wiki topics by session ids and expands cross-link endpoints", () => {
    const map = wikiMap();
    const topics = filterWikiTopicsWithCrossLinkClosure(map, new Set(["sess-1"]));
    const visibleIds = new Set(topics.map((item) => item.id));

    expect([...visibleIds].sort()).toEqual(["a", "c"]);
    expect(filterWikiEdges(map.tree_edges, visibleIds)).toEqual([]);
    expect(filterWikiEdges(map.cross_links, visibleIds).map((edge) => edge.id)).toEqual(["cross-ac"]);
  });

  it("resolves the cross-linked Session filter from high-confidence Wiki links only", () => {
    const map = wikiMap();
    map.cross_links.push({
      id: "cross-bc-low",
      snapshot_id: "snap-1",
      source: "b",
      target: "c",
      type: "shared_kg_evidence",
      evidence_type: "shared_kg_evidence",
      evidence_refs: [],
      attrs: { confidence: "low" },
    });

    const linked = highConfidenceCrossLinkedSessionIds(map);
    expect([...linked].sort()).toEqual(["sess-1", "sess-3"]);
    expect([
      ...resolveSessionFilterSelection({
        mode: "cross_linked",
        lanes: [lane("sess-1"), lane("sess-2"), lane("sess-3")],
        manualSelectedIds: new Set(),
        crossLinkedSessionIds: linked,
      }),
    ]).toEqual(["sess-1", "sess-3"]);
  });
});
