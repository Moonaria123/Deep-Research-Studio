import type { KnowledgeGraphNode, KnowledgeGraphSessionMap } from "@/lib/types";
import { extractSessionIdFromLane } from "./graphSessionFilter";

export interface SessionMapCrossLinkAnchor {
  memoryKey: string;
  fromSessionId: string;
  memoryFlowId: string;
  memoryNode: KnowledgeGraphNode;
}

export interface SessionMapCrossLinkEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  memoryKey: string;
  fromSessionId: string;
  toSessionId: string;
}

export interface SessionMapSharedAssetAnchor {
  assetKey: string;
  assetKind: "source" | "note" | "report" | "memory";
  fromSessionId: string;
  assetFlowId: string;
}

function addDirectedEdge(
  edges: SessionMapCrossLinkEdge[],
  seen: Set<string>,
  {
    id,
    source,
    target,
    label,
    memoryKey,
    fromSessionId,
    toSessionId,
  }: SessionMapCrossLinkEdge,
): void {
  if (seen.has(id)) return;
  seen.add(id);
  edges.push({ id, source, target, label, memoryKey, fromSessionId, toSessionId });
}

function addGroupedSharedAssetEdges(
  edges: SessionMapCrossLinkEdge[],
  seen: Set<string>,
  anchors: SessionMapSharedAssetAnchor[],
  sessionFlowIdBySessionId: ReadonlyMap<string, string>,
): void {
  const anchorsByKey = new Map<string, SessionMapSharedAssetAnchor[]>();
  anchors.forEach((anchor) => {
    anchorsByKey.set(anchor.assetKey, [...(anchorsByKey.get(anchor.assetKey) ?? []), anchor]);
  });

  anchorsByKey.forEach((group) => {
    const sessionIds = [...new Set(group.map((anchor) => anchor.fromSessionId))].filter((sessionId) =>
      sessionFlowIdBySessionId.has(sessionId),
    );
    if (sessionIds.length < 2) return;
    group.forEach((anchor) => {
      sessionIds.forEach((toSessionId) => {
        if (toSessionId === anchor.fromSessionId) return;
        const targetFlowId = sessionFlowIdBySessionId.get(toSessionId);
        if (!targetFlowId) return;
        addDirectedEdge(edges, seen, {
          id: `session-cross-${anchor.assetKey}-${anchor.fromSessionId}-${toSessionId}`,
          source: anchor.assetFlowId,
          target: targetFlowId,
          label: `cross-session shared ${anchor.assetKind}`,
          memoryKey: anchor.assetKey,
          fromSessionId: anchor.fromSessionId,
          toSessionId,
        });
      });
    });
  });
}

export function buildSessionMapCrossLinkEdges({
  memoryAnchors,
  sharedAnchors = [],
  nodesByKey,
  sessionFlowIdBySessionId,
}: {
  memoryAnchors: SessionMapCrossLinkAnchor[];
  sharedAnchors?: SessionMapSharedAssetAnchor[];
  nodesByKey: Record<string, KnowledgeGraphNode>;
  sessionFlowIdBySessionId: ReadonlyMap<string, string>;
}): SessionMapCrossLinkEdge[] {
  const edges: SessionMapCrossLinkEdge[] = [];
  const seen = new Set<string>();
  memoryAnchors.forEach((anchor) => {
    const references = anchor.memoryNode.referencing_sessions ?? nodesByKey[anchor.memoryKey]?.referencing_sessions ?? [];
    references.forEach((reference) => {
      const toSessionId = reference.session_id.trim();
      if (!toSessionId || toSessionId === anchor.fromSessionId) return;
      const targetFlowId = sessionFlowIdBySessionId.get(toSessionId);
      if (!targetFlowId) return;
      const id = `session-cross-${anchor.memoryKey}-${anchor.fromSessionId}-${toSessionId}`;
      addDirectedEdge(edges, seen, {
        id,
        source: anchor.memoryFlowId,
        target: targetFlowId,
        label: "cross-session memory reference",
        memoryKey: anchor.memoryKey,
        fromSessionId: anchor.fromSessionId,
        toSessionId,
      });
    });
  });
  addGroupedSharedAssetEdges(edges, seen, sharedAnchors, sessionFlowIdBySessionId);
  return edges;
}

export function sessionMapCrossLinkedSessionIds(sessionMap: KnowledgeGraphSessionMap | undefined): Set<string> {
  const linked = new Set<string>();
  if (!sessionMap) return linked;
  const available = new Set(sessionMap.sessions.map(extractSessionIdFromLane).filter(Boolean));
  const sourceSessions = new Map<string, Set<string>>();
  const noteSessions = new Map<string, Set<string>>();
  const reportSessions = new Map<string, Set<string>>();
  const memorySessions = new Map<string, Set<string>>();

  const addAssetSession = (byKey: Map<string, Set<string>>, assetKey: string, sessionId: string) => {
    if (!assetKey || !sessionId) return;
    byKey.set(assetKey, new Set([...(byKey.get(assetKey) ?? new Set<string>()), sessionId]));
  };

  sessionMap.sessions.forEach((lane) => {
    const sessionId = extractSessionIdFromLane(lane);
    lane.source_set.node_keys.forEach((sourceKey) => addAssetSession(sourceSessions, sourceKey, sessionId));
    lane.notes.forEach((note) => addAssetSession(noteSessions, note.key, sessionId));
    lane.reports.forEach((report) => addAssetSession(reportSessions, report.key, sessionId));
    lane.memory_set.node_keys.forEach((memoryKey) => {
      addAssetSession(memorySessions, memoryKey, sessionId);
      const memoryNode =
        lane.memory_set.members.find((node) => node.key === memoryKey) ?? sessionMap.nodes_by_key[memoryKey];
      memoryNode?.referencing_sessions?.forEach((reference) => {
        const toSessionId = reference.session_id.trim();
        if (toSessionId && toSessionId !== sessionId && available.has(toSessionId)) {
          linked.add(sessionId);
          linked.add(toSessionId);
        }
      });
    });
  });

  [sourceSessions, noteSessions, reportSessions, memorySessions].forEach((byKey) => {
    byKey.forEach((sessionIds) => {
      const existing = [...sessionIds].filter((sessionId) => available.has(sessionId));
      if (existing.length < 2) return;
      existing.forEach((sessionId) => linked.add(sessionId));
    });
  });

  return linked;
}
