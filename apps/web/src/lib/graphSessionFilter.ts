import type {
  KnowledgeGraphSessionLane,
  KnowledgeGraphWikiCrossLink,
  KnowledgeGraphWikiEdge,
  KnowledgeGraphWikiMap,
  KnowledgeGraphWikiTopic,
} from "@/lib/types";

export const DEFAULT_SESSION_FILTER_LIMIT = 8;

export type SessionFilterMode = "default" | "all" | "manual" | "cross_linked";

export interface SessionFilterOption {
  id: string;
  title: string;
}

function stringAttr(value: unknown): string {
  return typeof value === "string" ? value : "";
}

export function extractSessionIdFromLane(lane: KnowledgeGraphSessionLane): string {
  const attrId = stringAttr(lane.session.attrs?.session_id).trim();
  if (attrId) return attrId;
  const key = String(lane.session.key || "");
  return key.startsWith("session:") ? key.slice("session:".length) : key;
}

export function sessionFilterOptions(lanes: KnowledgeGraphSessionLane[]): SessionFilterOption[] {
  return lanes
    .map((lane) => ({
      id: extractSessionIdFromLane(lane),
      title: lane.session.title || extractSessionIdFromLane(lane) || "Untitled session",
    }))
    .filter((item) => item.id);
}

export function allSessionIds(lanes: KnowledgeGraphSessionLane[]): Set<string> {
  return new Set(sessionFilterOptions(lanes).map((item) => item.id));
}

export function defaultSelectedSessionIds(
  lanes: KnowledgeGraphSessionLane[],
  limit = DEFAULT_SESSION_FILTER_LIMIT,
): Set<string> {
  const ids = sessionFilterOptions(lanes).map((item) => item.id);
  if (ids.length <= limit) return new Set(ids);
  return new Set(ids.slice(0, limit));
}

export function resolveSessionFilterSelection({
  mode,
  lanes,
  manualSelectedIds,
  crossLinkedSessionIds,
  limit = DEFAULT_SESSION_FILTER_LIMIT,
}: {
  mode: SessionFilterMode;
  lanes: KnowledgeGraphSessionLane[];
  manualSelectedIds: Set<string>;
  crossLinkedSessionIds?: Set<string>;
  limit?: number;
}): Set<string> {
  if (mode === "all") return allSessionIds(lanes);
  if (mode === "cross_linked") {
    const available = allSessionIds(lanes);
    return new Set([...(crossLinkedSessionIds ?? new Set<string>())].filter((id) => available.has(id)));
  }
  if (mode === "manual") {
    const available = allSessionIds(lanes);
    return new Set([...manualSelectedIds].filter((id) => available.has(id)));
  }
  return defaultSelectedSessionIds(lanes, limit);
}

export function highConfidenceCrossLinkedSessionIds(wikiMap: KnowledgeGraphWikiMap | undefined): Set<string> {
  if (!wikiMap) return new Set();
  const byId = new Map(wikiMap.topics.map((topic) => [topic.id, topic]));
  const sessionIds = new Set<string>();
  wikiMap.cross_links.forEach((edge) => {
    if (stringAttr(edge.attrs?.confidence) !== "high") return;
    [byId.get(edge.source), byId.get(edge.target)].forEach((topic) => {
      topic?.session_ids.forEach((sessionId) => {
        if (sessionId) sessionIds.add(sessionId);
      });
    });
  });
  return sessionIds;
}

export function filterSessionLanes(
  lanes: KnowledgeGraphSessionLane[],
  selectedSessionIds: Set<string>,
): KnowledgeGraphSessionLane[] {
  return lanes.filter((lane) => selectedSessionIds.has(extractSessionIdFromLane(lane)));
}

export function filterWikiTopicsWithCrossLinkClosure(
  wikiMap: KnowledgeGraphWikiMap | undefined,
  selectedSessionIds: Set<string>,
): KnowledgeGraphWikiTopic[] {
  if (!wikiMap || selectedSessionIds.size === 0) return [];
  const byId = new Map(wikiMap.topics.map((topic) => [topic.id, topic]));
  const visibleIds = new Set(
    wikiMap.topics
      .filter((topic) => topic.session_ids.some((sessionId) => selectedSessionIds.has(sessionId)))
      .map((topic) => topic.id),
  );
  let changed = true;
  while (changed) {
    changed = false;
    wikiMap.cross_links.forEach((edge) => {
      const sourceVisible = visibleIds.has(edge.source);
      const targetVisible = visibleIds.has(edge.target);
      if (sourceVisible && !targetVisible && byId.has(edge.target)) {
        visibleIds.add(edge.target);
        changed = true;
      }
      if (targetVisible && !sourceVisible && byId.has(edge.source)) {
        visibleIds.add(edge.source);
        changed = true;
      }
    });
  }
  return wikiMap.topics.filter((topic) => visibleIds.has(topic.id));
}

export function filterWikiEdges<T extends KnowledgeGraphWikiEdge | KnowledgeGraphWikiCrossLink>(
  edges: T[],
  visibleTopicIds: Set<string>,
): T[] {
  return edges.filter((edge) => visibleTopicIds.has(edge.source) && visibleTopicIds.has(edge.target));
}
