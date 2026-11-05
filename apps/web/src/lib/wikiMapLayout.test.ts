import { describe, expect, it } from "vitest";
import { layoutWikiForest, normalizeWikiFocusTopicId } from "./wikiMapLayout";
import type { KnowledgeGraphWikiCrossLink, KnowledgeGraphWikiEdge, KnowledgeGraphWikiTopic } from "@/lib/types";

function topic(
  id: string,
  title: string,
  depth: number,
  parentId: string,
  sessionId: string,
  kind = "topic",
): KnowledgeGraphWikiTopic {
  return {
    id,
    snapshot_id: "snap-1",
    key: `wiki:${id}`,
    title,
    summary: `${title} summary`,
    depth,
    parent_id: parentId,
    session_ids: sessionId ? [sessionId] : [],
    source_message_ids: sessionId ? [`msg-${sessionId}`] : [],
    attrs: { kind, session_id: sessionId, session_title: `Session ${sessionId}` },
  };
}

const topics: KnowledgeGraphWikiTopic[] = [
  { ...topic("workspace", "Workspace", 0, "", "", "workspace_root"), attrs: { kind: "workspace_root" } },
  topic("s1-root", "Session 1", 1, "workspace", "s1", "session_root"),
  topic("s1-a", "Session 1 branch A", 2, "s1-root", "s1"),
  topic("s1-b", "Session 1 branch B", 2, "s1-root", "s1"),
  topic("s1-a-deep", "Session 1 deep idea", 3, "s1-a", "s1"),
  topic("s2-root", "Session 2", 1, "workspace", "s2", "session_root"),
  topic("s2-a", "Session 2 branch A", 2, "s2-root", "s2"),
  topic("s2-a-deep", "Session 2 deep idea", 3, "s2-a", "s2"),
];

const treeEdges: KnowledgeGraphWikiEdge[] = [
  { id: "e-w-s1", snapshot_id: "snap-1", source: "workspace", target: "s1-root", type: "parent_child", attrs: {} },
  { id: "e-s1-a", snapshot_id: "snap-1", source: "s1-root", target: "s1-a", type: "parent_child", attrs: {} },
  { id: "e-s1-b", snapshot_id: "snap-1", source: "s1-root", target: "s1-b", type: "parent_child", attrs: {} },
  { id: "e-s1-deep", snapshot_id: "snap-1", source: "s1-a", target: "s1-a-deep", type: "parent_child", attrs: {} },
  { id: "e-w-s2", snapshot_id: "snap-1", source: "workspace", target: "s2-root", type: "parent_child", attrs: {} },
  { id: "e-s2-a", snapshot_id: "snap-1", source: "s2-root", target: "s2-a", type: "parent_child", attrs: {} },
  { id: "e-s2-deep", snapshot_id: "snap-1", source: "s2-a", target: "s2-a-deep", type: "parent_child", attrs: {} },
];

const crossLinks: KnowledgeGraphWikiCrossLink[] = [
  {
    id: "cross-1",
    snapshot_id: "snap-1",
    source: "s1-a",
    target: "s2-a",
    type: "shared_kg_evidence",
    evidence_type: "shared_kg_evidence",
    evidence_refs: [{ node_key: "source:1" }],
    attrs: {},
  },
];

function layout(overrides: Partial<Parameters<typeof layoutWikiForest>[0]> = {}) {
  return layoutWikiForest({
    selectedTopics: topics,
    treeEdges,
    crossLinks,
    directSelectedTopicIds: new Set(topics.map((item) => item.id)),
    enabledTypes: new Set(["wikiTopic"]),
    query: "",
    showCrossLinks: false,
    expandedTopicIds: new Set(),
    ...overrides,
  });
}

describe("layoutWikiForest", () => {
  it("lays out each Session as a rightward tree and vertically packed forest", () => {
    const result = layout();
    const byId = new Map(result.nodes.map((node) => [node.id, node]));
    const root = byId.get("s1-root")!;
    const branchA = byId.get("s1-a")!;
    const branchB = byId.get("s1-b")!;
    const secondRoot = byId.get("s2-root")!;

    expect(branchA.x).toBeGreaterThan(root.x);
    expect(branchB.x).toBeGreaterThan(root.x);
    expect(branchA.y).not.toBe(branchB.y);
    expect(Math.abs(secondRoot.y - root.y)).toBeGreaterThan(120);
  });

  it("keeps L3 topics collapsed until their parent is expanded", () => {
    const collapsed = layout();
    expect(collapsed.nodes.map((node) => node.id)).not.toContain("s1-a-deep");
    expect(collapsed.nodes.find((node) => node.id === "s1-a")?.collapsedChildren).toBe(1);

    const expanded = layout({ expandedTopicIds: new Set(["s1-a"]) });
    expect(expanded.nodes.map((node) => node.id)).toContain("s1-a-deep");
  });

  it("hides cross-links by default and returns them when enabled", () => {
    expect(layout().crossEdges).toHaveLength(0);
    expect(layout({ showCrossLinks: true }).crossEdges.map((edge) => edge.id)).toEqual(["cross-1"]);
  });

  it("marks cross-link closure topics as related and gives them a related Session root", () => {
    const selectedTopics = topics.filter((item) => ["s1-root", "s1-a", "s2-a"].includes(item.id));
    const result = layout({
      selectedTopics,
      directSelectedTopicIds: new Set(["s1-root", "s1-a"]),
      showCrossLinks: false,
    });
    const relatedTopic = result.nodes.find((node) => node.id === "s2-a");
    const relatedRoot = result.nodes.find((node) => node.id === "wiki-related-session:s2");

    expect(relatedTopic?.related).toBe(true);
    expect(relatedRoot?.related).toBe(true);
    expect(result.crossEdges).toHaveLength(0);
  });

  it("focuses a branch while retaining its Session root", () => {
    const result = layout({ focusTopicId: "s1-a", expandedTopicIds: new Set(["s1-a"]) });
    expect(result.nodes.map((node) => node.id).sort()).toEqual(["s1-a", "s1-a-deep", "s1-root"]);
  });

  it("normalizes a deep focus target to its Session-level branch", () => {
    expect(normalizeWikiFocusTopicId("s1-a-deep", topics)).toBe("s1-a");
    expect(normalizeWikiFocusTopicId("s1-root", topics)).toBe("s1-root");
  });

  it("temporarily expands matching descendants with their ancestor chain", () => {
    const result = layout({ query: "deep idea" });

    expect(result.nodes.map((node) => node.id).sort()).toEqual([
      "s1-a",
      "s1-a-deep",
      "s1-root",
      "s2-a",
      "s2-a-deep",
      "s2-root",
    ]);
  });

  it("searches concept keys and temporarily suspends branch focus", () => {
    const selectedTopics = topics.map((item) =>
      item.id === "s2-a-deep" ? { ...item, attrs: { ...item.attrs, concept_key: "policy-forecast" } } : item,
    );
    const result = layout({ selectedTopics, focusTopicId: "s1-a", query: "policy-forecast" });

    expect(result.nodes.map((node) => node.id).sort()).toEqual(["s2-a", "s2-a-deep", "s2-root"]);
  });
});
