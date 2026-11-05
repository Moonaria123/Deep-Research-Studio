import { Graph, layout as dagreLayout } from "@dagrejs/dagre";
import type { KnowledgeGraphWikiCrossLink, KnowledgeGraphWikiEdge, KnowledgeGraphWikiTopic } from "@/lib/types";

const TOPIC_WIDTH = 248;
const TOPIC_HEIGHT = 96;
const SESSION_ROOT_WIDTH = 300;
const SESSION_ROOT_HEIGHT = 148;
const SESSION_GAP_Y = 160;

type TopicRole = "session_root" | "topic";

export interface WikiLayoutNode {
  id: string;
  topic: KnowledgeGraphWikiTopic;
  x: number;
  y: number;
  width: number;
  height: number;
  role: TopicRole;
  related: boolean;
  collapsedChildren: number;
}

export interface WikiLayoutEdge {
  id: string;
  source: string;
  target: string;
  edge: KnowledgeGraphWikiEdge | KnowledgeGraphWikiCrossLink;
}

export interface WikiForestLayout {
  nodes: WikiLayoutNode[];
  treeEdges: WikiLayoutEdge[];
  crossEdges: WikiLayoutEdge[];
  total: number;
}

export interface WikiForestLayoutInput {
  selectedTopics: KnowledgeGraphWikiTopic[];
  treeEdges: KnowledgeGraphWikiEdge[];
  crossLinks: KnowledgeGraphWikiCrossLink[];
  directSelectedTopicIds: Set<string>;
  enabledTypes: Set<string>;
  query: string;
  showCrossLinks: boolean;
  expandedTopicIds: Set<string>;
  focusTopicId?: string;
}

function stringAttr(topic: KnowledgeGraphWikiTopic, key: string): string {
  const value = topic.attrs?.[key];
  return typeof value === "string" ? value : "";
}

function topicSessionId(topic: KnowledgeGraphWikiTopic): string {
  return stringAttr(topic, "session_id") || topic.session_ids[0] || "unknown";
}

function isWorkspaceRoot(topic: KnowledgeGraphWikiTopic): boolean {
  return stringAttr(topic, "kind") === "workspace_root";
}

function isSessionRoot(topic: KnowledgeGraphWikiTopic): boolean {
  return stringAttr(topic, "kind") === "session_root";
}

function matchesQuery(topic: KnowledgeGraphWikiTopic, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  return [
    topic.title,
    topic.summary,
    topic.key,
    topicSessionId(topic),
    stringAttr(topic, "session_title"),
    stringAttr(topic, "concept_key"),
  ]
    .filter(Boolean)
    .some((value) => value.toLowerCase().includes(needle));
}

function makeSyntheticSessionRoot(sessionId: string, sample: KnowledgeGraphWikiTopic): KnowledgeGraphWikiTopic {
  const title = stringAttr(sample, "session_title") || `Related session ${sessionId}`;
  return {
    id: `wiki-related-session:${sessionId}`,
    snapshot_id: sample.snapshot_id,
    key: `wiki:related-session:${sessionId}`,
    title,
    summary: "Related topic from a cross-session evidence link.",
    depth: 1,
    parent_id: "",
    session_ids: [sessionId],
    source_message_ids: [],
    attrs: {
      kind: "session_root",
      session_id: sessionId,
      session_title: title,
      wiki_related: "true",
    },
  };
}

function descendantIds(rootId: string, childrenByParent: Map<string, KnowledgeGraphWikiTopic[]>): Set<string> {
  const result = new Set<string>();
  const stack = [...(childrenByParent.get(rootId) ?? [])];
  while (stack.length > 0) {
    const next = stack.pop()!;
    if (result.has(next.id)) continue;
    result.add(next.id);
    stack.push(...(childrenByParent.get(next.id) ?? []));
  }
  return result;
}

function ancestorIds(topic: KnowledgeGraphWikiTopic, byId: Map<string, KnowledgeGraphWikiTopic>): Set<string> {
  const result = new Set<string>();
  let cursor = byId.get(topic.parent_id);
  while (cursor) {
    result.add(cursor.id);
    cursor = byId.get(cursor.parent_id);
  }
  return result;
}

/** Normalize any topic to the first branch beneath its Session root. */
export function normalizeWikiFocusTopicId(
  topicId: string,
  topics: KnowledgeGraphWikiTopic[],
): string | undefined {
  const byId = new Map(topics.map((topic) => [topic.id, topic]));
  let cursor = byId.get(topicId);
  if (!cursor) return undefined;
  while (cursor.depth > 2) {
    const parent = byId.get(cursor.parent_id);
    if (!parent) break;
    cursor = parent;
  }
  return cursor.id;
}

function isVisibleByCollapse(topic: KnowledgeGraphWikiTopic, byId: Map<string, KnowledgeGraphWikiTopic>, expandedTopicIds: Set<string>): boolean {
  if (isSessionRoot(topic) || topic.depth <= 2) return true;
  let cursor = byId.get(topic.parent_id);
  while (cursor) {
    if (!isSessionRoot(cursor) && cursor.depth >= 2 && !expandedTopicIds.has(cursor.id)) return false;
    cursor = byId.get(cursor.parent_id);
  }
  return true;
}

function sortedTopics(topics: KnowledgeGraphWikiTopic[]): KnowledgeGraphWikiTopic[] {
  return [...topics].sort((left, right) => {
    const depthDelta = left.depth - right.depth;
    if (depthDelta !== 0) return depthDelta;
    return left.title.localeCompare(right.title) || left.id.localeCompare(right.id);
  });
}

function topicSize(topic: KnowledgeGraphWikiTopic): { width: number; height: number } {
  return isSessionRoot(topic) ? { width: SESSION_ROOT_WIDTH, height: SESSION_ROOT_HEIGHT } : { width: TOPIC_WIDTH, height: TOPIC_HEIGHT };
}

function layoutSessionGroup(
  topics: KnowledgeGraphWikiTopic[],
  groupEdges: KnowledgeGraphWikiEdge[],
  directSelectedTopicIds: Set<string>,
  collapsedChildrenById: Map<string, number>,
  yOffset: number,
): { nodes: WikiLayoutNode[]; edges: WikiLayoutEdge[]; height: number } {
  const graph = new Graph({ directed: true, multigraph: true });
  graph.setGraph({ rankdir: "LR", nodesep: 64, ranksep: 104, marginx: 0, marginy: 0 });
  graph.setDefaultEdgeLabel(() => ({}));

  sortedTopics(topics).forEach((topic) => {
    graph.setNode(topic.id, topicSize(topic));
  });
  groupEdges
    .slice()
    .sort((left, right) => left.id.localeCompare(right.id))
    .forEach((edge) => graph.setEdge(edge.source, edge.target, {}, edge.id));

  dagreLayout(graph);

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  const nodes = sortedTopics(topics).map((topic) => {
    const value = graph.node(topic.id) as { x: number; y: number; width: number; height: number };
    minX = Math.min(minX, value.x - value.width / 2);
    minY = Math.min(minY, value.y - value.height / 2);
    maxY = Math.max(maxY, value.y + value.height / 2);
    return { topic, value };
  });

  const normalizedNodes = nodes.map(({ topic, value }) => ({
    id: topic.id,
    topic,
    x: Math.round(value.x - value.width / 2 - minX),
    y: Math.round(value.y - value.height / 2 - minY + yOffset),
    width: value.width,
    height: value.height,
    role: (isSessionRoot(topic) ? "session_root" : "topic") as TopicRole,
    related: !directSelectedTopicIds.has(topic.id) || stringAttr(topic, "wiki_related") === "true",
    collapsedChildren: collapsedChildrenById.get(topic.id) ?? 0,
  }));

  const edgeIds = new Set(groupEdges.map((edge) => edge.id));
  const normalizedEdges = graph
    .edges()
    .filter((edge) => edgeIds.has(edge.name ?? ""))
    .map((edge) => {
      const sourceEdge = groupEdges.find((item) => item.id === edge.name)!;
      return { id: sourceEdge.id, source: sourceEdge.source, target: sourceEdge.target, edge: sourceEdge };
    });

  return { nodes: normalizedNodes, edges: normalizedEdges, height: Math.max(0, maxY - minY) };
}

export function layoutWikiForest({
  selectedTopics,
  treeEdges,
  crossLinks,
  directSelectedTopicIds,
  enabledTypes,
  query,
  showCrossLinks,
  expandedTopicIds,
  focusTopicId,
}: WikiForestLayoutInput): WikiForestLayout {
  if (!enabledTypes.has("wikiTopic")) return { nodes: [], treeEdges: [], crossEdges: [], total: selectedTopics.length };

  const nonWorkspaceTopics = selectedTopics.filter((topic) => !isWorkspaceRoot(topic));
  const byId = new Map(nonWorkspaceTopics.map((topic) => [topic.id, topic]));
  const childrenByParent = new Map<string, KnowledgeGraphWikiTopic[]>();
  treeEdges.forEach((edge) => {
    const child = byId.get(edge.target);
    if (!child || !byId.has(edge.source)) return;
    childrenByParent.set(edge.source, [...(childrenByParent.get(edge.source) ?? []), child]);
  });

  const normalizedFocusTopicId = query.trim() ? undefined : normalizeWikiFocusTopicId(focusTopicId ?? "", nonWorkspaceTopics);
  let focusIds: Set<string> | undefined;
  if (normalizedFocusTopicId && byId.has(normalizedFocusTopicId)) {
    const focusTopic = byId.get(normalizedFocusTopicId)!;
    focusIds = new Set([focusTopic.id, ...ancestorIds(focusTopic, byId), ...descendantIds(focusTopic.id, childrenByParent)]);
  }

  const queryVisibleIds = new Set<string>();
  if (query.trim()) {
    nonWorkspaceTopics.filter((topic) => matchesQuery(topic, query)).forEach((topic) => {
      queryVisibleIds.add(topic.id);
      ancestorIds(topic, byId).forEach((id) => queryVisibleIds.add(id));
    });
  }

  const visibleCandidates = nonWorkspaceTopics.filter((topic) => {
    if (focusIds && !focusIds.has(topic.id)) return false;
    if (query.trim()) return queryVisibleIds.has(topic.id);
    if (!isVisibleByCollapse(topic, byId, expandedTopicIds)) return false;
    return true;
  });
  const visibleIds = new Set(visibleCandidates.map((topic) => topic.id));

  const collapsedChildrenById = new Map<string, number>();
  visibleCandidates.forEach((topic) => {
    const hiddenCount = [...descendantIds(topic.id, childrenByParent)].filter((id) => !visibleIds.has(id)).length;
    if (hiddenCount > 0) collapsedChildrenById.set(topic.id, hiddenCount);
  });

  const groups = new Map<string, KnowledgeGraphWikiTopic[]>();
  visibleCandidates.forEach((topic) => {
    const sessionId = topicSessionId(topic);
    groups.set(sessionId, [...(groups.get(sessionId) ?? []), topic]);
  });

  const normalizedGroups = [...groups.entries()].map(([sessionId, topics]) => {
    const root = topics.find(isSessionRoot) ?? makeSyntheticSessionRoot(sessionId, topics[0]);
    const groupTopics = topics.some((topic) => topic.id === root.id) ? topics : [root, ...topics];
    return { sessionId, root, topics: groupTopics };
  });

  let yOffset = 0;
  const nodes: WikiLayoutNode[] = [];
  const layoutTreeEdges: WikiLayoutEdge[] = [];

  normalizedGroups
    .sort((left, right) => left.root.title.localeCompare(right.root.title) || left.sessionId.localeCompare(right.sessionId))
    .forEach((group) => {
      const topicIds = new Set(group.topics.map((topic) => topic.id));
      const groupEdges = treeEdges.filter((edge) => topicIds.has(edge.source) && topicIds.has(edge.target));
      group.topics.forEach((topic) => {
        if (topic.id === group.root.id) return;
        const hasVisibleParent = groupEdges.some((edge) => edge.target === topic.id);
        if (!hasVisibleParent) {
          groupEdges.push({
            id: `synthetic-${group.root.id}-${topic.id}`,
            snapshot_id: topic.snapshot_id,
            source: group.root.id,
            target: topic.id,
            type: "parent_child",
            attrs: { synthetic: true },
          });
        }
      });
      const laidOut = layoutSessionGroup(group.topics, groupEdges, directSelectedTopicIds, collapsedChildrenById, yOffset);
      nodes.push(...laidOut.nodes);
      layoutTreeEdges.push(...laidOut.edges);
      yOffset += laidOut.height + SESSION_GAP_Y;
    });

  const finalVisibleIds = new Set(nodes.map((node) => node.id));
  const layoutCrossEdges = showCrossLinks
    ? crossLinks
        .filter((edge) => finalVisibleIds.has(edge.source) && finalVisibleIds.has(edge.target))
        .map((edge) => ({ id: edge.id, source: edge.source, target: edge.target, edge }))
    : [];

  return { nodes, treeEdges: layoutTreeEdges, crossEdges: layoutCrossEdges, total: selectedTopics.length };
}
