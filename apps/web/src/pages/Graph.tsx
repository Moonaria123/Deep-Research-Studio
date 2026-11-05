import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  applyNodeChanges,
  Background,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  MarkerType,
  Position,
  type Edge as FlowEdge,
  type NodeChange,
  type NodeTypes,
  type Viewport,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Loader2, Lock, Maximize2, RefreshCw, RotateCcw, Search, ServerCrash, Unlock, Workflow } from "lucide-react";
import { api, ApiError, streamKnowledgeGraphJob } from "@/lib/api";
import { resolveGraphJumpTarget } from "@/lib/graphJump";
import {
  buildSessionMapCrossLinkEdges,
  sessionMapCrossLinkedSessionIds,
  type SessionMapCrossLinkAnchor,
  type SessionMapSharedAssetAnchor,
} from "@/lib/sessionMapCrossLinks";
import { clearSessionMapState, readSessionMapCrossLinks, writeSessionMapCrossLinks } from "@/lib/sessionMapState";
import { layoutWikiForest, normalizeWikiFocusTopicId, type WikiLayoutNode } from "@/lib/wikiMapLayout";
import {
  clearWikiSnapshotState,
  readWikiBoolean,
  readWikiString,
  readWikiStringSet,
  writeWikiBoolean,
  writeWikiString,
  writeWikiStringSet,
} from "@/lib/wikiMapState";
import type {
  KnowledgeGraphGraph,
  KnowledgeGraphBuildJob,
  KnowledgeGraphJobEvent,
  KnowledgeGraphNode,
  KnowledgeGraphNodeDetail,
  KnowledgeGraphSessionLane,
  KnowledgeGraphSessionMap,
  KnowledgeGraphSessionRef,
  KnowledgeGraphStatus,
  KnowledgeGraphWikiMap,
  KnowledgeGraphWikiCrossLink,
  KnowledgeGraphWikiRefreshJob,
  KnowledgeGraphWikiTopic,
} from "@/lib/types";
import { cn, formatTime } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { GraphCanvasOverlayStack, type GraphCanvasOverlayItem } from "@/components/graph/GraphCanvasOverlayStack";
import { GraphPageHeader } from "@/components/graph/GraphPageHeader";
import { GraphToolbar } from "@/components/graph/GraphToolbar";
import { GraphViewTabs, type KnowledgeGraphViewMode } from "@/components/graph/GraphViewTabs";
import { HubNode } from "@/components/graph/HubNode";
import { MemoryMemberNode } from "@/components/graph/MemoryMemberNode";
import { NoteMemberNode } from "@/components/graph/NoteMemberNode";
import { SessionFilter } from "@/components/graph/SessionFilter";
import { SessionMapControls } from "@/components/graph/SessionMapControls";
import { SessionNode } from "@/components/graph/SessionNode";
import { WikiMapControls } from "@/components/graph/WikiMapControls";
import { findWikiTopicForNode, WikiInspector } from "@/components/graph/WikiInspector";
import { WikiSessionRootNode } from "@/components/graph/WikiSessionRootNode";
import { WikiTopicNode } from "@/components/graph/WikiTopicNode";
import { summarizeWikiPartial } from "@/lib/wikiPartialSummary";
import { wikiErrorCodeHint, wikiNoCrossLinkHint } from "@/lib/wikiOperatorHints";
import { IS_DEMO } from "@/lib/runtime";
import {
  filterSessionLanes,
  filterWikiTopicsWithCrossLinkClosure,
  extractSessionIdFromLane,
  highConfidenceCrossLinkedSessionIds,
  resolveSessionFilterSelection,
  sessionFilterOptions,
  type SessionFilterMode,
} from "@/lib/graphSessionFilter";
import {
  colorFor,
  GraphNode,
  stringAttr,
  type GraphFlowNode,
  type GraphNodeData,
  typeLabel,
} from "@/components/graph/GraphNode";

const WORKSPACE_ID = "default";
const NODE_LIMIT = 300;
const SESSION_LIMIT = 50;
const MEMBER_PREVIEW = 100;
const SESSION_LAYOUT_MODE: KnowledgeGraphViewMode = "session_tree";
const NODE_WIDTH = 248;
const SESSION_WIDTH = 320;
const HUB_WIDTH = 132;
const SOURCE_MEMBER_X = 1080;
const MEMORY_MEMBER_X = SOURCE_MEMBER_X + NODE_WIDTH + 232;
const SOURCE_HUB_OFFSET_Y = 20;
const MEMORY_HUB_OFFSET_Y = 130;
const ACTIVE_JOB_STATUSES = new Set(["queued", "running"]);
const CONCEPT_EDGE_TYPES = new Set([
  "concept_mentioned_in_note",
  "concept_supported_by_source",
  "concept_summarized_by_report",
  "concept_remembered_in_memory",
]);
type SessionFilterState = {
  mode: SessionFilterMode;
  manualSelectedSessionIds: Set<string>;
};
const GRAPH_NODE_TYPES: NodeTypes = {
  session: GraphNode,
  source: GraphNode,
  memory: MemoryMemberNode,
  report: GraphNode,
  note: NoteMemberNode,
  sessionCard: SessionNode,
  sourceHub: HubNode,
  memoryHub: HubNode,
  wikiSessionRoot: WikiSessionRootNode,
  wikiTopic: WikiTopicNode,
  concept: GraphNode,
};

function isActiveJob(job?: { status?: string } | null): boolean {
  return typeof job?.status === "string" && ACTIVE_JOB_STATUSES.has(job.status);
}

function jobEventNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function mergeBuildJobEvent(
  job: KnowledgeGraphBuildJob,
  event: KnowledgeGraphJobEvent,
): KnowledgeGraphBuildJob {
  const payload = event.payload;
  return {
    ...job,
    status: typeof payload.status === "string" ? payload.status : job.status,
    stage: typeof payload.stage === "string" ? payload.stage : job.stage,
    progress: jobEventNumber(payload.progress) ?? job.progress,
    error_code: typeof payload.error_code === "string" ? payload.error_code : job.error_code,
    stats_json: payload.stats && typeof payload.stats === "object" ? payload.stats : job.stats_json,
  };
}

function mergeWikiJobEvent(
  job: KnowledgeGraphWikiRefreshJob,
  event: KnowledgeGraphJobEvent,
): KnowledgeGraphWikiRefreshJob {
  const payload = event.payload;
  return {
    ...job,
    status: typeof payload.status === "string" ? payload.status : job.status,
    stage: typeof payload.stage === "string" ? payload.stage : job.stage,
    progress: jobEventNumber(payload.progress) ?? job.progress,
    attempted_sessions: jobEventNumber(payload.attempted_sessions) ?? job.attempted_sessions,
    succeeded_sessions: jobEventNumber(payload.succeeded_sessions) ?? job.succeeded_sessions,
    skipped_sessions: jobEventNumber(payload.skipped_sessions) ?? job.skipped_sessions,
    failed_sessions: jobEventNumber(payload.failed_sessions) ?? job.failed_sessions,
    error_code: typeof payload.error_code === "string" ? payload.error_code : job.error_code,
    stats: payload.stats && typeof payload.stats === "object"
      ? { ...job.stats, ...payload.stats }
      : job.stats,
  };
}

function wikiRefreshJobErrorCode(job?: KnowledgeGraphWikiRefreshJob | null): string {
  if (job?.status !== "failed") return "";
  if (typeof job.error_code === "string" && job.error_code) return job.error_code;
  const stats = job.stats ?? {};
  if (typeof stats.error_code === "string" && stats.error_code) return stats.error_code;
  const firstError = Array.isArray(stats.errors) ? stats.errors[0] : undefined;
  if (firstError && typeof firstError === "object" && typeof (firstError as { error_code?: unknown }).error_code === "string") {
    return (firstError as { error_code: string }).error_code;
  }
  return "";
}

function formatPercent(value?: number | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  return `${Math.round(value * 100)}%`;
}

function graphVersionId(graph?: KnowledgeGraphGraph, sessionMap?: KnowledgeGraphSessionMap): string {
  return sessionMap?.active_version?.id || graph?.active_version?.id || "none";
}

function wikiVersionId(wikiMap?: KnowledgeGraphWikiMap): string {
  return wikiMap?.active_snapshot?.id || "none";
}

function layoutStoragePrefix(workspaceId: string, layoutMode: KnowledgeGraphViewMode, versionId: string): string {
  return `kg-layout:${workspaceId}:${layoutMode}:${versionId}:`;
}

function layoutStorageKey(workspaceId: string, layoutMode: KnowledgeGraphViewMode, versionId: string, nodeKey: string): string {
  return `${layoutStoragePrefix(workspaceId, layoutMode, versionId)}${nodeKey}`;
}

function viewportStorageKey(workspaceId: string, layoutMode: KnowledgeGraphViewMode, versionId: string): string {
  return `kg-viewport:${workspaceId}:${layoutMode}:${versionId}`;
}

function safeLocalStorage(): Storage | null {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    return null;
  }
}

function readStoredPosition(
  workspaceId: string,
  layoutMode: KnowledgeGraphViewMode,
  versionId: string,
  nodeKey: string,
): { x: number; y: number } | null {
  const storage = safeLocalStorage();
  if (!storage) return null;
  try {
    const raw = storage.getItem(layoutStorageKey(workspaceId, layoutMode, versionId, nodeKey));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { x?: unknown; y?: unknown };
    if (typeof parsed.x !== "number" || typeof parsed.y !== "number") return null;
    return { x: parsed.x, y: parsed.y };
  } catch {
    return null;
  }
}

function writeStoredPosition(
  workspaceId: string,
  layoutMode: KnowledgeGraphViewMode,
  versionId: string,
  nodeKey: string,
  position: { x: number; y: number },
) {
  const storage = safeLocalStorage();
  if (!storage) return;
  storage.setItem(layoutStorageKey(workspaceId, layoutMode, versionId, nodeKey), JSON.stringify(position));
}

function readStoredViewport(
  workspaceId: string,
  layoutMode: KnowledgeGraphViewMode,
  versionId: string,
): Viewport | null {
  const storage = safeLocalStorage();
  if (!storage) return null;
  try {
    const raw = storage.getItem(viewportStorageKey(workspaceId, layoutMode, versionId));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { x?: unknown; y?: unknown; zoom?: unknown };
    if (typeof parsed.x !== "number" || typeof parsed.y !== "number" || typeof parsed.zoom !== "number") return null;
    return { x: parsed.x, y: parsed.y, zoom: parsed.zoom };
  } catch {
    return null;
  }
}

function writeStoredViewport(
  workspaceId: string,
  layoutMode: KnowledgeGraphViewMode,
  versionId: string,
  viewport: Viewport,
) {
  const storage = safeLocalStorage();
  if (!storage) return;
  storage.setItem(viewportStorageKey(workspaceId, layoutMode, versionId), JSON.stringify(viewport));
}

function clearStoredLayout(workspaceId: string, layoutMode: KnowledgeGraphViewMode, versionId: string) {
  const storage = safeLocalStorage();
  if (!storage) return;
  const positionPrefix = layoutStoragePrefix(workspaceId, layoutMode, versionId);
  const viewportKey = viewportStorageKey(workspaceId, layoutMode, versionId);
  for (let index = storage.length - 1; index >= 0; index -= 1) {
    const key = storage.key(index);
    if (key?.startsWith(positionPrefix) || key === viewportKey) storage.removeItem(key);
  }
}


function looksLikeRawMemoryTitle(node: KnowledgeGraphNode, title: string): boolean {
  if (!title) return false;
  const memoryId = stringAttr(node, "memory_id");
  return title === node.key || title === memoryId || /^(memory|mem)[:\s_-]/i.test(title);
}

function looksLikeRawNoteTitle(node: KnowledgeGraphNode, title: string): boolean {
  if (!title) return false;
  const noteId = stringAttr(node, "note_id");
  return title === node.key || title === noteId || /^(note|mem)[:\s_-]/i.test(title);
}

function graphNodeLabel(node: KnowledgeGraphNode): string {
  if (node.type === "memory") {
    if (node.summary) return node.summary;
    if (node.title && !looksLikeRawMemoryTitle(node, node.title)) return node.title;
    return "Memory";
  }
  if (node.type === "note") {
    if (node.summary) return node.summary;
    if (node.title && !looksLikeRawNoteTitle(node, node.title)) return node.title;
    return "Note";
  }
  return node.title || node.key;
}

function flowNode(
  node: KnowledgeGraphNode,
  position: { x: number; y: number },
  layoutMode: KnowledgeGraphViewMode,
  versionId: string,
  flowType?: string,
  extraData?: Partial<GraphNodeData>,
  flowId?: string,
): GraphFlowNode {
  const storedPosition = readStoredPosition(WORKSPACE_ID, layoutMode, versionId, node.key);
  return {
    id: flowId ?? (flowType ? `${flowType}:${node.key}` : node.id),
    type: flowType ?? node.type,
    position: storedPosition ?? position,
    sourcePosition: Position.Right,
    targetPosition: Position.Left,
    data: {
      label: graphNodeLabel(node),
      graphNode: node,
      ...extraData,
    },
    style: { width: flowType === "sessionCard" || flowType === "wikiSessionRoot" ? SESSION_WIDTH : flowType?.endsWith("Hub") ? HUB_WIDTH : NODE_WIDTH },
  };
}

function syntheticHubNode(lane: KnowledgeGraphSessionLane, kind: "source" | "memory"): KnowledgeGraphNode {
  const sessionKey = lane.session.key;
  const count = kind === "source" ? lane.source_set.count : lane.memory_set.count;
  return {
    id: `hub-${kind}-${sessionKey}`,
    type: `${kind}Hub`,
    key: `hub:${kind}:${sessionKey}`,
    title: `${kind === "source" ? "Sources" : "Memories"} ${count}`,
    summary: "",
    confidence: 1,
    visible: true,
    attrs: {
      hub_kind: kind,
      session_key: sessionKey,
    },
  };
}

function matchesQuery(node: KnowledgeGraphNode, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  return [node.title, node.summary, node.key, node.type, stringAttr(node, "assistant_summary")]
    .filter(Boolean)
    .some((value) => String(value).toLowerCase().includes(needle));
}

function laneMatchesQuery(lane: KnowledgeGraphSessionLane, sessionMap: KnowledgeGraphSessionMap, query: string): boolean {
  if (!query.trim()) return true;
  const members = [...lane.reports, ...lane.notes, ...lane.source_set.node_keys.map((key) => sessionMap.nodes_by_key[key]), ...lane.memory_set.node_keys.map((key) => sessionMap.nodes_by_key[key])].filter(Boolean) as KnowledgeGraphNode[];
  return [lane.session, ...members].some((node) => matchesQuery(node, query));
}

function sessionMapLayout(
  sessionMap: KnowledgeGraphSessionMap | undefined,
  {
    versionId,
    expandedHubs,
    expandedMemories,
    expandedNotes,
    expandedSessions,
    enabledTypes,
    selectedSessionIds,
    query,
    showCrossLinks,
  }: {
    versionId: string;
    expandedHubs: Set<string>;
    expandedMemories: Set<string>;
    expandedNotes: Set<string>;
    expandedSessions: Set<string>;
    enabledTypes: Set<string>;
    selectedSessionIds: Set<string>;
    query: string;
    showCrossLinks: boolean;
  },
): { nodes: GraphFlowNode[]; edges: FlowEdge[]; total: number } {
  if (!sessionMap) return { nodes: [], edges: [], total: 0 };
  const nodes: GraphFlowNode[] = [];
  const edges: FlowEdge[] = [];
  const memoryAnchors: SessionMapCrossLinkAnchor[] = [];
  const sharedAnchors: SessionMapSharedAssetAnchor[] = [];
  const sessionFlowIdBySessionId = new Map<string, string>();
  let y = 0;
  const lanes = filterSessionLanes(sessionMap.sessions, selectedSessionIds);
  lanes.forEach((lane, laneIndex) => {
    if (!enabledTypes.has("session") || !laneMatchesQuery(lane, sessionMap, query)) return;
    const sessionId = extractSessionIdFromLane(lane);
    const sourceExpanded = expandedHubs.has(`source:${lane.session.key}`);
    const memoryExpanded = expandedHubs.has(`memory:${lane.session.key}`);
    const sessionExpanded = expandedSessions.has(lane.session.key);
    const sourceMembers = lane.source_set.node_keys.map((key) => sessionMap.nodes_by_key[key]).filter(Boolean) as KnowledgeGraphNode[];
    const memoryMembers = lane.memory_set.node_keys.map((key) => sessionMap.nodes_by_key[key]).filter(Boolean) as KnowledgeGraphNode[];
    const visibleSources = sourceExpanded && enabledTypes.has("source") ? sourceMembers : [];
    const visibleMemories = memoryExpanded && enabledTypes.has("memory") ? memoryMembers : [];
    const visibleReports = enabledTypes.has("report") ? lane.reports : [];
    const visibleNotes = enabledTypes.has("note") ? lane.notes : [];
    const laneY = y;
    const sessionFlow = flowNode(
      lane.session,
      { x: 0, y: laneY },
      SESSION_LAYOUT_MODE,
      versionId,
      "sessionCard",
      { expanded: sessionExpanded, counts: lane.counts },
    );
    nodes.push(sessionFlow);
    if (sessionId) sessionFlowIdBySessionId.set(sessionId, sessionFlow.id);

    const assetColumn = [...visibleReports, ...visibleNotes];
    assetColumn.forEach((node, index) => {
      const assetFlow = flowNode(
        node,
        { x: 420, y: laneY + (index - (assetColumn.length - 1) / 2) * 120 },
        SESSION_LAYOUT_MODE,
        versionId,
        undefined,
        node.type === "note" ? { expanded: expandedNotes.has(node.key) } : undefined,
        `${node.type}:${lane.session.key}:${node.key}`,
      );
      nodes.push(assetFlow);
      if (sessionId && (node.type === "note" || node.type === "report")) {
        sharedAnchors.push({
          assetKey: node.key,
          assetKind: node.type,
          fromSessionId: sessionId,
          assetFlowId: assetFlow.id,
        });
      }
      edges.push(mapEdge(`session-asset-${lane.session.key}-${node.key}`, sessionFlow.id, assetFlow.id, node.type === "note" ? "session note" : "session report"));
    });

    const sourceHub = syntheticHubNode(lane, "source");
    const memoryHub = syntheticHubNode(lane, "memory");
    const hubFlows: Partial<Record<"source" | "memory", GraphFlowNode>> = {};
    const hubPairs: [KnowledgeGraphNode, "source" | "memory", number, boolean][] = [
      [sourceHub, "source", SOURCE_HUB_OFFSET_Y, sourceExpanded],
      [memoryHub, "memory", MEMORY_HUB_OFFSET_Y, memoryExpanded],
    ];
    hubPairs.forEach(([hub, kind, offsetY, expanded]) => {
      const typeAllowed = enabledTypes.has(kind);
      if (!typeAllowed || Number(hub.title.replace(/\D/g, "") || 0) <= 0) return;
      const hubFlow = flowNode(hub, { x: 760, y: laneY + offsetY }, SESSION_LAYOUT_MODE, versionId, `${kind}Hub`, {
        hubKind: kind,
        hubCount: kind === "source" ? lane.source_set.count : lane.memory_set.count,
        expanded,
      });
      nodes.push(hubFlow);
      hubFlows[kind] = hubFlow;
      edges.push(mapEdge(`session-hub-${lane.session.key}-${kind}`, sessionFlow.id, hubFlow.id, kind === "source" ? "session sources" : "session memories"));
    });

    visibleSources.forEach((node, index) => {
      const hubId = hubFlows.source?.id;
      if (!hubId) return;
      const itemFlow = flowNode(
        node,
        { x: SOURCE_MEMBER_X, y: laneY + SOURCE_HUB_OFFSET_Y + index * 106 },
        SESSION_LAYOUT_MODE,
        versionId,
        undefined,
        undefined,
        `source:${lane.session.key}:${node.key}`,
      );
      nodes.push(itemFlow);
      if (sessionId) {
        sharedAnchors.push({
          assetKey: node.key,
          assetKind: "source",
          fromSessionId: sessionId,
          assetFlowId: itemFlow.id,
        });
      }
      edges.push(mapEdge(`source-member-${lane.session.key}-${node.key}`, hubId, itemFlow.id, "source member"));
    });
    visibleMemories.forEach((node, index) => {
      const hubId = hubFlows.memory?.id;
      if (!hubId) return;
      const itemFlow = flowNode(
        node,
        { x: MEMORY_MEMBER_X, y: laneY + MEMORY_HUB_OFFSET_Y + index * 106 },
        SESSION_LAYOUT_MODE,
        versionId,
        undefined,
        { expanded: expandedMemories.has(node.key) },
        `memory:${lane.session.key}:${node.key}`,
      );
      nodes.push(itemFlow);
      if (sessionId) {
        memoryAnchors.push({
          memoryKey: node.key,
          fromSessionId: sessionId,
          memoryFlowId: itemFlow.id,
          memoryNode: node,
        });
        sharedAnchors.push({
          assetKey: node.key,
          assetKind: "memory",
          fromSessionId: sessionId,
          assetFlowId: itemFlow.id,
        });
      }
      edges.push(mapEdge(`memory-member-${lane.session.key}-${node.key}`, hubId, itemFlow.id, "memory member", true));
    });

    const sourceSpan = visibleSources.length ? SOURCE_HUB_OFFSET_Y + visibleSources.length * 106 : 0;
    const memorySpan = visibleMemories.length ? MEMORY_HUB_OFFSET_Y + visibleMemories.length * 106 : 0;
    const assetSpan = assetColumn.length * 120;
    const verticalSpan = Math.max(220, sourceSpan, memorySpan, assetSpan);
    y += 260 + verticalSpan + (sessionExpanded ? 120 : 0) + (laneIndex === lanes.length - 1 ? 0 : 36);
  });
  if (showCrossLinks) {
    buildSessionMapCrossLinkEdges({
      memoryAnchors,
      sharedAnchors,
      nodesByKey: sessionMap.nodes_by_key,
      sessionFlowIdBySessionId,
    }).forEach((edge) => {
      edges.push(
        mapEdge(edge.id, edge.source, edge.target, edge.label, true, {
          stroke: "#64748b",
          opacity: 0.45,
          zIndex: -1,
        }),
      );
    });
  }
  return { nodes, edges, total: Object.keys(sessionMap.nodes_by_key).length };
}

function withConceptOverlay(
  base: { nodes: GraphFlowNode[]; edges: FlowEdge[]; total: number },
  graph: KnowledgeGraphGraph | undefined,
  {
    enabledTypes,
    query,
    versionId,
  }: {
    enabledTypes: Set<string>;
    query: string;
    versionId: string;
  },
): { nodes: GraphFlowNode[]; edges: FlowEdge[]; total: number } {
  const conceptTotal = graph?.nodes.filter((node) => node.type === "concept").length ?? 0;
  if (!graph || !enabledTypes.has("concept")) {
    return { ...base, total: base.total + conceptTotal };
  }
  const conceptsById = new Map(
    graph.nodes.filter((node) => node.type === "concept").map((node) => [node.id, node]),
  );
  const visibleTargets = new Map<string, GraphFlowNode[]>();
  base.nodes.forEach((node) => {
    const graphId = node.data.graphNode.id;
    const current = visibleTargets.get(graphId) ?? [];
    current.push(node);
    visibleTargets.set(graphId, current);
  });
  const edgesByConcept = new Map<string, typeof graph.edges>();
  graph.edges.forEach((edge) => {
    if (!CONCEPT_EDGE_TYPES.has(edge.type) || !conceptsById.has(edge.source)) return;
    const current = edgesByConcept.get(edge.source) ?? [];
    current.push(edge);
    edgesByConcept.set(edge.source, current);
  });

  const conceptNodes: GraphFlowNode[] = [];
  const conceptEdges: FlowEdge[] = [];
  [...conceptsById.values()]
    .sort((left, right) => left.key.localeCompare(right.key))
    .slice(0, 64)
    .forEach((concept, index) => {
      const targetFlows = (edgesByConcept.get(concept.id) ?? [])
        .flatMap((edge) => visibleTargets.get(edge.target) ?? [])
        .filter((node, targetIndex, all) => all.findIndex((item) => item.id === node.id) === targetIndex);
      if (targetFlows.length === 0) return;
      if (query.trim() && !matchesQuery(concept, query) && !targetFlows.some((node) => matchesQuery(node.data.graphNode, query))) {
        return;
      }
      const anchor = targetFlows[0];
      const conceptFlow = flowNode(
        concept,
        {
          x: Math.max(...targetFlows.map((node) => node.position.x)) + NODE_WIDTH + 180,
          y: anchor.position.y + index * 24,
        },
        SESSION_LAYOUT_MODE,
        versionId,
      );
      conceptNodes.push(conceptFlow);
      targetFlows.forEach((target) => {
        const edge = (edgesByConcept.get(concept.id) ?? []).find((item) => item.target === target.data.graphNode.id);
        if (!edge) return;
        conceptEdges.push(
          mapEdge(
            `concept-${edge.id}-${target.id}`,
            conceptFlow.id,
            target.id,
            edge.type.replaceAll("_", " "),
          ),
        );
      });
    });
  return {
    nodes: [...base.nodes, ...conceptNodes],
    edges: [...base.edges, ...conceptEdges],
    total: base.total + conceptTotal,
  };
}

function wikiTopicGraphNode(topic: KnowledgeGraphWikiTopic, layoutNode?: WikiLayoutNode): KnowledgeGraphNode {
  const kind = typeof topic.attrs?.kind === "string" ? topic.attrs.kind : "";
  const role = layoutNode?.role ?? (kind === "session_root" ? "session_root" : "topic");
  const related = layoutNode?.related || topic.attrs?.wiki_related === "true";
  const collapsedChildren = layoutNode?.collapsedChildren ?? 0;
  return {
    id: topic.id,
    type: "wikiTopic",
    key: topic.key,
    title: topic.title,
    summary: topic.summary,
    confidence: 1,
    visible: true,
    attrs: {
      ...topic.attrs,
      depth: String(topic.depth),
      session_ids: topic.session_ids.join(", "),
      source_message_ids: topic.source_message_ids.join(", "),
      color_index: String(Math.max(0, topic.depth % 6)),
      wiki_node_role: role,
      wiki_related: related ? "true" : "",
      collapsed_children: collapsedChildren > 0 ? String(collapsedChildren) : "",
    },
  };
}

function wikiMapLayout(
  wikiMap: KnowledgeGraphWikiMap | undefined,
  {
    versionId,
    enabledTypes,
    selectedSessionIds,
    query,
    showCrossLinks,
    expandedTopicIds,
    focusTopicId,
  }: {
    versionId: string;
    enabledTypes: Set<string>;
    selectedSessionIds: Set<string>;
    query: string;
    showCrossLinks: boolean;
    expandedTopicIds: Set<string>;
    focusTopicId?: string;
  },
): { nodes: GraphFlowNode[]; edges: FlowEdge[]; total: number } {
  if (!wikiMap || !enabledTypes.has("wikiTopic")) return { nodes: [], edges: [], total: wikiMap?.topics.length ?? 0 };
  const selectedTopics = filterWikiTopicsWithCrossLinkClosure(wikiMap, selectedSessionIds);
  const directSelectedTopicIds = new Set(
    wikiMap.topics
      .filter((topic) => topic.session_ids.some((sessionId) => selectedSessionIds.has(sessionId)))
      .map((topic) => topic.id),
  );
  const layout = layoutWikiForest({
    selectedTopics,
    treeEdges: wikiMap.tree_edges,
    crossLinks: wikiMap.cross_links,
    directSelectedTopicIds,
    enabledTypes,
    query,
    showCrossLinks,
    expandedTopicIds,
    focusTopicId,
  });
  const nodes = layout.nodes.map((layoutNode) => {
    const graphNode = wikiTopicGraphNode(layoutNode.topic, layoutNode);
    return flowNode(
      graphNode,
      { x: layoutNode.x, y: layoutNode.y },
      "wiki_map",
      versionId,
      layoutNode.role === "session_root" ? "wikiSessionRoot" : "wikiTopic",
      undefined,
      layoutNode.topic.id,
    );
  });
  const edges = [
    ...layout.treeEdges.map((edge) => mapEdge(`wiki-tree-${edge.id}`, edge.source, edge.target, "topic branch")),
    ...layout.crossEdges.map((edge) =>
      mapEdge(`wiki-cross-${edge.id}`, edge.source, edge.target, edge.edge.type || "evidence link", true, {
        stroke: "#64748b",
        opacity: 0.45,
        zIndex: -1,
      }),
    ),
  ];
  return { nodes, edges, total: layout.total };
}

function mapEdge(
  id: string,
  source: string,
  target: string,
  label: string,
  dashed = false,
  options: { stroke?: string; opacity?: number; zIndex?: number } = {},
): FlowEdge {
  const stroke = options.stroke ?? (dashed ? "#e11d48" : "#475569");
  return {
    id,
    source,
    target,
    type: "smoothstep",
    ariaLabel: label,
    zIndex: options.zIndex ?? 0,
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: stroke,
      width: 18,
      height: 18,
    },
    style: {
      stroke,
      strokeWidth: dashed ? 2 : 1.75,
      strokeDasharray: dashed ? "6 4" : undefined,
      opacity: options.opacity ?? (dashed ? 0.9 : 0.85),
    },
  };
}

function nodeAttrRows(node?: KnowledgeGraphNode | KnowledgeGraphNodeDetail): [string, string][] {
  if (!node?.attrs) return [];
  return Object.entries(node.attrs)
    .filter(([, value]) => typeof value === "string" && value.trim())
    .slice(0, 8) as [string, string][];
}

function memoryReverseLinks(node?: KnowledgeGraphNode | KnowledgeGraphNodeDetail): KnowledgeGraphSessionRef[] {
  if (!node || node.type !== "memory") return [];
  return node.referencing_sessions ?? [];
}

function GraphFlowCanvas({
  nodes,
  edges,
  locked,
  fitViewRequest,
  viewMode,
  versionId,
  onNodesChange,
  onNodeClick,
  onNodeDoubleClick,
  onEdgeClick,
  onNodeMouseEnter,
  onNodeMouseLeave,
}: {
  nodes: GraphFlowNode[];
  edges: FlowEdge[];
  locked: boolean;
  fitViewRequest: number;
  viewMode: KnowledgeGraphViewMode;
  versionId: string;
  onNodesChange: (changes: NodeChange<GraphFlowNode>[]) => void;
  onNodeClick: (node: KnowledgeGraphNode) => void;
  onNodeDoubleClick: (node: KnowledgeGraphNode) => void;
  onEdgeClick?: (edgeId: string) => void;
  onNodeMouseEnter: (nodeId: string) => void;
  onNodeMouseLeave: () => void;
}) {
  const { fitView, getViewport, setViewport } = useReactFlow();
  const hasGraphNodes = nodes.length > 0;
  const restoredViewportKeyRef = useRef<string | undefined>(undefined);

  useEffect(() => {
    const viewportKey = `${viewMode}:${versionId}`;
    if (!hasGraphNodes || restoredViewportKeyRef.current === viewportKey) return;
    restoredViewportKeyRef.current = viewportKey;
    const stored = readStoredViewport(WORKSPACE_ID, viewMode, versionId);
    if (stored) void setViewport(stored, { duration: 0 });
  }, [hasGraphNodes, setViewport, versionId, viewMode]);

  useEffect(() => {
    if (!fitViewRequest || !hasGraphNodes) return undefined;
    const timer = window.setTimeout(() => {
      void fitView({ padding: 0.18, duration: 160 });
    }, 0);
    return () => window.clearTimeout(timer);
  }, [fitView, fitViewRequest, hasGraphNodes]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={GRAPH_NODE_TYPES}
      defaultEdgeOptions={{
        type: "smoothstep",
        markerEnd: { type: MarkerType.ArrowClosed, width: 18, height: 18 },
      }}
      elevateEdgesOnSelect={false}
      onNodesChange={onNodesChange}
      nodesDraggable={!locked}
      nodesConnectable={!locked}
      panOnDrag={!locked}
      zoomOnScroll={!locked}
      zoomOnPinch={!locked}
      zoomOnDoubleClick={!locked}
      autoPanOnNodeDrag={!locked}
      elementsSelectable
      minZoom={0.2}
      maxZoom={1.7}
      proOptions={{ hideAttribution: true }}
      onMoveEnd={() => {
        if (!locked) {
          writeStoredViewport(WORKSPACE_ID, viewMode, versionId, getViewport());
        }
      }}
      onNodeClick={(_, node) => onNodeClick(node.data.graphNode)}
      onNodeMouseEnter={(_, node) => onNodeMouseEnter(node.id)}
      onNodeMouseLeave={onNodeMouseLeave}
      onNodeDoubleClick={(_, node) => onNodeDoubleClick(node.data.graphNode)}
      onEdgeClick={(_, edge) => onEdgeClick?.(edge.id)}
    >
      <Background gap={22} color="rgba(100, 116, 139, 0.25)" />
      <MiniMap nodeColor={(node) => colorFor((node.data as GraphNodeData).graphNode.type).minimap} pannable={!locked} zoomable={!locked} />
    </ReactFlow>
  );
}

export function Graph() {
  const navigate = useNavigate();
  const [status, setStatus] = useState<KnowledgeGraphStatus>();
  const [graph, setGraph] = useState<KnowledgeGraphGraph>();
  const [sessionMap, setSessionMap] = useState<KnowledgeGraphSessionMap>();
  const [wikiMap, setWikiMap] = useState<KnowledgeGraphWikiMap>();
  const [wikiJob, setWikiJob] = useState<KnowledgeGraphWikiRefreshJob | null>(null);
  const [viewMode, setViewMode] = useState<KnowledgeGraphViewMode>(SESSION_LAYOUT_MODE);
  const [positionedNodes, setPositionedNodes] = useState<GraphFlowNode[]>([]);
  const [selectedNode, setSelectedNode] = useState<KnowledgeGraphNode | KnowledgeGraphNodeDetail>();
  const [hoveredNodeId, setHoveredNodeId] = useState<string>();
  const [viewportLocked, setViewportLocked] = useState(false);
  const [fitViewRequest, setFitViewRequest] = useState(0);
  const [layoutResetNonce, setLayoutResetNonce] = useState(0);
  const [query, setQuery] = useState("");
  const [enabledTypes, setEnabledTypes] = useState<Set<string>>(new Set());
  const [sessionMapFilter, setSessionMapFilter] = useState<SessionFilterState>({
    mode: "default",
    manualSelectedSessionIds: new Set(),
  });
  const [wikiMapFilter, setWikiMapFilter] = useState<SessionFilterState>({
    mode: "default",
    manualSelectedSessionIds: new Set(),
  });
  const [expandedHubs, setExpandedHubs] = useState<Set<string>>(new Set());
  const [expandedMemories, setExpandedMemories] = useState<Set<string>>(new Set());
  const [expandedNotes, setExpandedNotes] = useState<Set<string>>(new Set());
  const [expandedSessions, setExpandedSessions] = useState<Set<string>>(new Set());
  const [expandedWikiTopics, setExpandedWikiTopics] = useState<Set<string>>(new Set());
  const [focusedWikiTopicId, setFocusedWikiTopicId] = useState<string>();
  const [selectedWikiCrossLink, setSelectedWikiCrossLink] = useState<KnowledgeGraphWikiCrossLink | null>(null);
  const [showSessionCrossLinks, setShowSessionCrossLinks] = useState(false);
  const [showWikiCrossLinks, setShowWikiCrossLinks] = useState(false);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const [wikiRefreshing, setWikiRefreshing] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [rebuildError, setRebuildError] = useState("");
  const [wikiRefreshError, setWikiRefreshError] = useState("");
  const jobEventCursorsRef = useRef(new Map<string, number>());
  const reconciledTerminalJobsRef = useRef(new Set<string>());

  const loadGraph = useCallback(async () => {
    const [nextGraph, nextSessionMap, nextWikiMap] = await Promise.all([
      api.getKnowledgeGraph({ workspace_id: WORKSPACE_ID, limit: NODE_LIMIT }),
      api.getKnowledgeGraphSessionMap({ workspace_id: WORKSPACE_ID, session_limit: SESSION_LIMIT, member_preview: MEMBER_PREVIEW }),
      api.getKnowledgeGraphWikiMap({ workspace_id: WORKSPACE_ID }),
    ]);
    setGraph(nextGraph);
    setSessionMap(nextSessionMap);
    setWikiMap(nextWikiMap);
    setWikiJob(nextWikiMap.last_refresh_job);
    setFitViewRequest((current) => current + 1);
    setEnabledTypes((current) => {
      if (current.size > 0) {
        if (nextWikiMap.topics.length && !current.has("wikiTopic")) {
          return new Set([...current, "wikiTopic"]);
        }
        return current;
      }
      return new Set([
        ...nextGraph.nodes.filter((node) => node.type !== "concept").map((node) => node.type),
        ...Object.values(nextSessionMap.nodes_by_key).map((node) => node.type),
        ...(nextWikiMap.topics.length ? ["wikiTopic"] : []),
      ]);
    });
  }, []);

  const loadStatus = useCallback(async () => {
    const nextStatus = await api.getKnowledgeGraphStatus({ workspace_id: WORKSPACE_ID });
    setStatus(nextStatus);
    return nextStatus;
  }, []);

  const loadAll = useCallback(async () => {
    setLoadError("");
    setLoading(true);
    try {
      await Promise.all([loadStatus(), loadGraph()]);
    } catch (err) {
      setLoadError(err instanceof ApiError ? err.message : "Could not load the knowledge graph.");
    } finally {
      setLoading(false);
    }
  }, [loadGraph, loadStatus]);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  useEffect(() => {
    const job = status?.last_build_job;
    if (!job?.id || !isActiveJob(job)) return undefined;
    const jobId = job.id;
    const jobKey = `graph:${jobId}`;
    let disposed = false;
    let terminalSeen = false;
    let stopStream: (() => void) | undefined;
    let reconnectTimer: number | undefined;

    const reconcileTerminal = () => {
      if (reconciledTerminalJobsRef.current.has(jobKey)) return;
      reconciledTerminalJobsRef.current.add(jobKey);
      void loadStatus().then(() => loadGraph());
    };
    const connect = () => {
      if (disposed || terminalSeen) return;
      const cursor = jobEventCursorsRef.current.get(jobKey) ?? 0;
      stopStream = streamKnowledgeGraphJob(
        "graph",
        jobId,
        WORKSPACE_ID,
        (event: KnowledgeGraphJobEvent) => {
          const previous = jobEventCursorsRef.current.get(jobKey) ?? 0;
          if (event.seq <= previous) return;
          jobEventCursorsRef.current.set(jobKey, event.seq);
          setStatus((current) => {
            if (!current?.last_build_job || current.last_build_job.id !== jobId) {
              return current;
            }
            return {
              ...current,
              last_build_job: mergeBuildJobEvent(current.last_build_job, event),
            };
          });
          if (event.type === "job_terminal") {
            terminalSeen = true;
            reconcileTerminal();
          }
        },
        () => {
          if (!disposed && !terminalSeen) {
            reconnectTimer = window.setTimeout(connect, 50);
          }
        },
        () => {
          if (!disposed && !terminalSeen) {
            reconnectTimer = window.setTimeout(connect, 500);
          }
        },
        cursor,
      );
    };

    connect();
    return () => {
      disposed = true;
      stopStream?.();
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
    };
  }, [loadGraph, loadStatus, status?.last_build_job?.id, status?.last_build_job?.status]);

  useEffect(() => {
    const jobId = wikiJob?.id;
    if (!jobId || !isActiveJob(wikiJob)) return undefined;
    const jobKey = `wiki:${jobId}`;
    let disposed = false;
    let terminalSeen = false;
    let stopStream: (() => void) | undefined;
    let reconnectTimer: number | undefined;

    const reconcileTerminal = () => {
      if (reconciledTerminalJobsRef.current.has(jobKey)) return;
      reconciledTerminalJobsRef.current.add(jobKey);
      void api
        .getKnowledgeGraphWikiMapJob(jobId, { workspace_id: WORKSPACE_ID })
        .then((nextJob) => {
          setWikiJob(nextJob);
          return loadGraph();
        });
    };
    const connect = () => {
      if (disposed || terminalSeen) return;
      const cursor = jobEventCursorsRef.current.get(jobKey) ?? 0;
      stopStream = streamKnowledgeGraphJob(
        "wiki",
        jobId,
        WORKSPACE_ID,
        (event: KnowledgeGraphJobEvent) => {
          const previous = jobEventCursorsRef.current.get(jobKey) ?? 0;
          if (event.seq <= previous) return;
          jobEventCursorsRef.current.set(jobKey, event.seq);
          setWikiJob((current) => (
            current?.id === jobId ? mergeWikiJobEvent(current, event) : current
          ));
          if (event.type === "job_terminal") {
            terminalSeen = true;
            reconcileTerminal();
          }
        },
        () => {
          if (!disposed && !terminalSeen) {
            reconnectTimer = window.setTimeout(connect, 50);
          }
        },
        () => {
          if (!disposed && !terminalSeen) {
            reconnectTimer = window.setTimeout(connect, 500);
          }
        },
        cursor,
      );
    };

    connect();
    return () => {
      disposed = true;
      stopStream?.();
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
    };
  }, [loadGraph, wikiJob?.id, wikiJob?.status]);

  const sessionVersionId = graphVersionId(graph, sessionMap);
  const currentLayoutMode = viewMode;
  const currentWikiSnapshotId = wikiVersionId(wikiMap);
  const versionId = viewMode === "wiki_map" ? currentWikiSnapshotId : sessionVersionId;

  useEffect(() => {
    setShowSessionCrossLinks(readSessionMapCrossLinks(WORKSPACE_ID, sessionVersionId));
  }, [sessionVersionId]);

  useEffect(() => {
    if (!wikiMap?.active_snapshot) {
      setExpandedWikiTopics(new Set());
      setFocusedWikiTopicId(undefined);
      setShowWikiCrossLinks(false);
      return;
    }
    setExpandedWikiTopics(readWikiStringSet("expanded", WORKSPACE_ID, currentWikiSnapshotId));
    const storedFocus = readWikiString("focus", WORKSPACE_ID, currentWikiSnapshotId);
    setFocusedWikiTopicId(storedFocus && wikiMap.topics.some((topic) => topic.id === storedFocus) ? storedFocus : undefined);
    setShowWikiCrossLinks(readWikiBoolean("cross-links", WORKSPACE_ID, currentWikiSnapshotId));
  }, [currentWikiSnapshotId, wikiMap?.active_snapshot, wikiMap?.topics]);
  const nodeTypes = useMemo(() => {
    const types = [
      ...(graph?.nodes.map((node) => node.type) ?? []),
      ...Object.values(sessionMap?.nodes_by_key ?? {}).map((node) => node.type),
      ...(wikiMap?.topics.length ? ["wikiTopic"] : []),
    ];
    const uniqueTypes = [...new Set(types)];
    return uniqueTypes.sort((a, b) => typeLabel(a).localeCompare(typeLabel(b)));
  }, [graph?.nodes, sessionMap?.nodes_by_key, wikiMap?.topics.length]);
  const sessionOptions = useMemo(() => sessionFilterOptions(sessionMap?.sessions ?? []), [sessionMap?.sessions]);
  const sessionMapLinkedSessionIds = useMemo(() => sessionMapCrossLinkedSessionIds(sessionMap), [sessionMap]);
  const wikiLinkedSessionIds = useMemo(() => highConfidenceCrossLinkedSessionIds(wikiMap), [wikiMap]);
  const sessionMapSelectedSessionIds = useMemo(
    () =>
      resolveSessionFilterSelection({
        mode: sessionMapFilter.mode,
        lanes: sessionMap?.sessions ?? [],
        manualSelectedIds: sessionMapFilter.manualSelectedSessionIds,
        crossLinkedSessionIds: sessionMapLinkedSessionIds,
      }),
    [sessionMapFilter.manualSelectedSessionIds, sessionMapFilter.mode, sessionMapLinkedSessionIds, sessionMap?.sessions],
  );
  const wikiSelectedSessionIds = useMemo(
    () =>
      resolveSessionFilterSelection({
        mode: wikiMapFilter.mode,
        lanes: sessionMap?.sessions ?? [],
        manualSelectedIds: wikiMapFilter.manualSelectedSessionIds,
        crossLinkedSessionIds: wikiLinkedSessionIds,
      }),
    [sessionMap?.sessions, wikiLinkedSessionIds, wikiMapFilter.manualSelectedSessionIds, wikiMapFilter.mode],
  );
  const activeSessionFilter = viewMode === "wiki_map" ? wikiMapFilter : sessionMapFilter;
  const effectiveSelectedSessionIds = viewMode === "wiki_map" ? wikiSelectedSessionIds : sessionMapSelectedSessionIds;

  const sessionLayout = useMemo(
    () => {
      const base = sessionMapLayout(sessionMap, {
        versionId: sessionVersionId,
        expandedHubs,
        expandedMemories,
        expandedNotes,
        expandedSessions,
        enabledTypes,
        selectedSessionIds: sessionMapSelectedSessionIds,
        query,
        showCrossLinks: showSessionCrossLinks,
      });
      return withConceptOverlay(base, graph, {
        enabledTypes,
        query,
        versionId: sessionVersionId,
      });
    },
    [
      enabledTypes,
      expandedHubs,
      expandedMemories,
      expandedNotes,
      expandedSessions,
      layoutResetNonce,
      query,
      graph,
      sessionMap,
      sessionMapSelectedSessionIds,
      sessionVersionId,
      showSessionCrossLinks,
    ],
  );

  const wikiLayout = useMemo(
    () =>
      wikiMapLayout(wikiMap, {
        versionId: currentWikiSnapshotId,
        enabledTypes,
        selectedSessionIds: wikiSelectedSessionIds,
        query,
        showCrossLinks: showWikiCrossLinks,
        expandedTopicIds: expandedWikiTopics,
        focusTopicId: query.trim() ? undefined : focusedWikiTopicId,
      }),
    [
      currentWikiSnapshotId,
      enabledTypes,
      expandedWikiTopics,
      focusedWikiTopicId,
      layoutResetNonce,
      query,
      showWikiCrossLinks,
      wikiMap,
      wikiSelectedSessionIds,
    ],
  );

  const activeLayout = viewMode === "wiki_map" ? wikiLayout : sessionLayout;
  const baseFlowNodes = activeLayout.nodes;
  const flowEdges = activeLayout.edges;

  useEffect(() => {
    setPositionedNodes(baseFlowNodes);
  }, [baseFlowNodes]);

  const displayNodes = useMemo(
    () =>
      positionedNodes.map((node) => ({
        ...node,
        data: {
          ...node.data,
          selected: selectedNode?.id === node.data.graphNode.id,
          highlighted: hoveredNodeId === node.id,
        },
      })),
    [hoveredNodeId, positionedNodes, selectedNode?.id],
  );
  const visibleWikiTopicIds = useMemo(
    () => new Set(displayNodes.filter((node) => node.data.graphNode.type === "wikiTopic").map((node) => node.data.graphNode.id)),
    [displayNodes],
  );
  const activeJob = isActiveJob(status?.last_build_job);
  const activeWikiJob = isActiveJob(wikiJob);
  const empty =
    !loading &&
    !loadError &&
    !activeJob &&
    (viewMode === "wiki_map" ? (wikiMap?.topics.length ?? 0) === 0 : (sessionMap?.sessions.length ?? 0) === 0);
  const filteredEmpty = !loading && !loadError && !activeJob && !empty && activeLayout.nodes.length === 0;
  const manualSelectionMissing =
    activeSessionFilter.mode === "manual" &&
    activeSessionFilter.manualSelectedSessionIds.size > 0 &&
    effectiveSelectedSessionIds.size === 0;

  const toggleType = (type: string) => {
    setEnabledTypes((current) => {
      const next = new Set(current);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  };

  const toggleHub = (node: KnowledgeGraphNode) => {
    const kind = stringAttr(node, "hub_kind");
    const sessionKey = stringAttr(node, "session_key");
    if (!kind || !sessionKey) return;
    const key = `${kind}:${sessionKey}`;
    setExpandedHubs((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const toggleWikiTopic = (node: KnowledgeGraphNode) => {
    setExpandedWikiTopics((current) => {
      const next = new Set(current);
      if (next.has(node.id)) next.delete(node.id);
      else next.add(node.id);
      writeWikiStringSet("expanded", WORKSPACE_ID, currentWikiSnapshotId, next);
      return next;
    });
  };

  const toggleWikiCrossLinks = () => {
    setShowWikiCrossLinks((current) => {
      const next = !current;
      writeWikiBoolean("cross-links", WORKSPACE_ID, currentWikiSnapshotId, next);
      return next;
    });
  };

  const toggleSessionCrossLinks = () => {
    setShowSessionCrossLinks((current) => {
      const next = !current;
      writeSessionMapCrossLinks(WORKSPACE_ID, sessionVersionId, next);
      return next;
    });
  };

  const selectNode = async (node: KnowledgeGraphNode) => {
    setSelectedWikiCrossLink(null);
    if (node.type === "sourceHub" || node.type === "memoryHub") {
      toggleHub(node);
      return;
    }
    if (node.type === "memory") {
      setExpandedMemories((current) => {
        const next = new Set(current);
        if (next.has(node.key)) next.delete(node.key);
        else next.add(node.key);
        return next;
      });
    }
    if (node.type === "note") {
      setExpandedNotes((current) => {
        const next = new Set(current);
        if (next.has(node.key)) next.delete(node.key);
        else next.add(node.key);
        return next;
      });
    }
    if (node.type === "session") {
      setExpandedSessions((current) => {
        const next = new Set(current);
        if (next.has(node.key)) next.delete(node.key);
        else next.add(node.key);
        return next;
      });
    }
    if (node.type === "wikiTopic") {
      toggleWikiTopic(node);
      setSelectedNode(node);
      return;
    }
    const referencingSessions = node.referencing_sessions;
    setSelectedNode(node);
    setDetailLoading(true);
    try {
      const detail = await api.getKnowledgeGraphNode(node.id);
      setSelectedNode({ ...detail.node, referencing_sessions: referencingSessions });
    } catch {
      setSelectedNode(node);
    } finally {
      setDetailLoading(false);
    }
  };

  const selectWikiTopicById = useCallback(
    (topicId: string) => {
      const topic = wikiMap?.topics.find((item) => item.id === topicId);
      if (topic) {
        setSelectedWikiCrossLink(null);
        setSelectedNode(wikiTopicGraphNode(topic));
      }
    },
    [wikiMap],
  );

  const focusWikiTopic = useCallback(
    (topicId: string) => {
      const normalized = normalizeWikiFocusTopicId(topicId, wikiMap?.topics ?? []);
      if (!normalized) return;
      setFocusedWikiTopicId(normalized);
      writeWikiString("focus", WORKSPACE_ID, currentWikiSnapshotId, normalized);
    },
    [currentWikiSnapshotId, wikiMap?.topics],
  );

  const clearWikiFocus = useCallback(() => {
    setFocusedWikiTopicId(undefined);
    writeWikiString("focus", WORKSPACE_ID, currentWikiSnapshotId, undefined);
  }, [currentWikiSnapshotId]);

  const selectWikiCrossLink = useCallback(
    (edgeId: string) => {
      if (viewMode !== "wiki_map" || !edgeId.startsWith("wiki-cross-")) return;
      const crossLinkId = edgeId.slice("wiki-cross-".length);
      const crossLink = wikiMap?.cross_links.find((edge) => edge.id === crossLinkId);
      if (!crossLink) return;
      setSelectedWikiCrossLink(crossLink);
      const sourceTopic = wikiMap?.topics.find((topic) => topic.id === crossLink.source);
      if (sourceTopic) setSelectedNode(wikiTopicGraphNode(sourceTopic));
    },
    [viewMode, wikiMap],
  );

  const handleNodesChange = (changes: NodeChange<GraphFlowNode>[]) => {
    if (viewportLocked) return;
    setPositionedNodes((currentNodes) => {
      const nextNodes = applyNodeChanges(changes, currentNodes) as GraphFlowNode[];
      changes.forEach((change) => {
        if (change.type !== "position" || !change.position || change.dragging) return;
        const changedNode = nextNodes.find((node) => node.id === change.id);
        if (changedNode) {
          writeStoredPosition(WORKSPACE_ID, currentLayoutMode, versionId, changedNode.data.graphNode.key, changedNode.position);
        }
      });
      return nextNodes;
    });
  };

  const resetLayout = () => {
    clearStoredLayout(WORKSPACE_ID, currentLayoutMode, versionId);
    if (currentLayoutMode === "wiki_map") {
      clearWikiSnapshotState(WORKSPACE_ID, currentWikiSnapshotId);
      setExpandedWikiTopics(new Set());
      setFocusedWikiTopicId(undefined);
      setSelectedWikiCrossLink(null);
      setShowWikiCrossLinks(false);
    } else {
      clearSessionMapState(WORKSPACE_ID, sessionVersionId);
      setShowSessionCrossLinks(false);
    }
    setLayoutResetNonce((current) => current + 1);
    setFitViewRequest((current) => current + 1);
  };

  const requestFitView = () => {
    setFitViewRequest((current) => current + 1);
  };

  const updateActiveSessionFilter = (updater: (current: SessionFilterState) => SessionFilterState) => {
    if (viewMode === "wiki_map") setWikiMapFilter(updater);
    else setSessionMapFilter(updater);
  };

  const showAllSessions = () => {
    updateActiveSessionFilter((current) => ({ ...current, mode: "all" }));
    setFitViewRequest((current) => current + 1);
  };

  const showCrossLinkedSessions = () => {
    updateActiveSessionFilter(() => ({ mode: "cross_linked", manualSelectedSessionIds: new Set() }));
    setFitViewRequest((current) => current + 1);
  };

  const clearSessionFilter = () => {
    updateActiveSessionFilter(() => ({ mode: "manual", manualSelectedSessionIds: new Set() }));
    setFitViewRequest((current) => current + 1);
  };

  const resetSessionFilter = () => {
    updateActiveSessionFilter(() => ({ mode: "default", manualSelectedSessionIds: new Set() }));
    setFitViewRequest((current) => current + 1);
  };

  const toggleSessionFilter = (sessionId: string) => {
    const selectedSnapshot = new Set(effectiveSelectedSessionIds);
    updateActiveSessionFilter((current) => {
      const next = new Set(current.mode === "manual" ? current.manualSelectedSessionIds : selectedSnapshot);
      if (next.has(sessionId)) next.delete(sessionId);
      else next.add(sessionId);
      return { mode: "manual", manualSelectedSessionIds: next };
    });
    setFitViewRequest((current) => current + 1);
  };

  const changeViewMode = (mode: KnowledgeGraphViewMode) => {
    setViewMode(mode);
    setSelectedNode(undefined);
    setSelectedWikiCrossLink(null);
    setHoveredNodeId(undefined);
    setFitViewRequest((current) => current + 1);
  };

  const rebuild = async () => {
    setRebuilding(true);
    setRebuildError("");
    try {
      const job = await api.rebuildKnowledgeGraph({ workspace_id: WORKSPACE_ID });
      if (job) {
        setStatus((current) => ({
          workspace_id: WORKSPACE_ID,
          active_version: current?.active_version ?? null,
          counts: current?.counts ?? { nodes: 0, edges: 0 },
          last_build_job: job,
        }));
      }
    } catch (err) {
      if (err instanceof ApiError && err.code === "KNOWLEDGE_GRAPH_BUILD_RUNNING") {
        setRebuildError("A graph rebuild is already running. This page will keep checking for updates.");
        void loadStatus();
      } else {
        setRebuildError(err instanceof ApiError ? err.message : "Could not start graph rebuild.");
      }
    } finally {
      setRebuilding(false);
    }
  };

  const refreshWiki = async () => {
    setWikiRefreshing(true);
    setWikiRefreshError("");
    try {
      const job = await api.refreshKnowledgeGraphWikiMap({ workspace_id: WORKSPACE_ID });
      setWikiJob(job);
    } catch (err) {
      if (err instanceof ApiError && err.code === "KNOWLEDGE_GRAPH_WIKI_REFRESH_RUNNING") {
        setWikiRefreshError("A Wiki Map refresh is already running. This page will keep checking for updates.");
        void loadGraph();
      } else {
        setWikiRefreshError(err instanceof ApiError ? err.message : "Could not start Wiki Map refresh.");
      }
    } finally {
      setWikiRefreshing(false);
    }
  };

  const detailSummary = selectedNode
    ? selectedNode.summary ||
      stringAttr(selectedNode, "content_excerpt") ||
      stringAttr(selectedNode, "assistant_summary") ||
      stringAttr(selectedNode, "assistant_excerpt")
    : "";
  const reverseLinks = memoryReverseLinks(selectedNode);
  const selectedWikiTopic = findWikiTopicForNode(selectedNode, wikiMap);
  const shownCount = displayNodes.length;
  const totalCount = activeLayout.total;
  const wikiPartial = summarizeWikiPartial(wikiMap);
  const wikiPartialErrorHint = wikiErrorCodeHint(wikiPartial?.dominantErrorCode);
  const wikiJobErrorCode = wikiRefreshJobErrorCode(wikiJob);
  const wikiJobErrorHint = wikiErrorCodeHint(wikiJobErrorCode);
  const visibleCrossEdgeCount = flowEdges.filter((edge) =>
    viewMode === "wiki_map" ? edge.id.startsWith("wiki-cross-") : edge.id.startsWith("session-cross-"),
  ).length;
  const crossLinkNotice =
    viewMode === SESSION_LAYOUT_MODE && showSessionCrossLinks && visibleCrossEdgeCount === 0
      ? {
          title: "No cross-session links available for the current Session filter.",
          detail: "Shared source or memory links require matching visible Sessions and expanded members.",
        }
      : viewMode === "wiki_map" && showWikiCrossLinks && visibleCrossEdgeCount === 0
        ? wikiNoCrossLinkHint({ wikiMap, visibleCrossEdgeCount }) ?? undefined
        : undefined;
  const graphStatusBadges = (
    <>
      <Badge variant={activeJob ? "warning" : status?.last_build_job?.status === "failed" ? "danger" : "accent"}>
        {activeJob
          ? `Building · ${status?.last_build_job?.stage ?? "queued"}`
          : status?.active_version
            ? `${status.counts.nodes} nodes · ${status.counts.edges} edges`
            : "No active graph"}
      </Badge>
      {wikiMap?.active_snapshot ? (
        <Badge variant={wikiMap.stale ? "warning" : wikiMap.active_snapshot.status === "partial" ? "warning" : "accent"}>
          {wikiMap.stale
            ? "Wiki stale"
            : wikiMap.active_snapshot.status === "partial"
              ? "Wiki partial"
              : `${wikiMap.stats.topics ?? wikiMap.topics.length} wiki topics`}
        </Badge>
      ) : null}
    </>
  );
  const graphActions = (
    <>
      <Button variant="outline" size="sm" onClick={() => void loadAll()} disabled={loading}>
        {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
        <span className="max-[520px]:sr-only">Refresh</span>
      </Button>
      <Button size="sm" onClick={() => void rebuild()} disabled={IS_DEMO || rebuilding || activeJob}>
        {rebuilding ? <Loader2 className="h-4 w-4 animate-spin" /> : <Workflow className="h-4 w-4" />}
        <span className="max-[520px]:sr-only">Rebuild graph</span>
      </Button>
      <Button
        variant="outline"
        size="sm"
        onClick={() => void refreshWiki()}
        disabled={IS_DEMO || wikiRefreshing || activeWikiJob || !status?.active_version}
      >
        {wikiRefreshing || activeWikiJob ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
        <span className="max-[520px]:sr-only">Refresh Wiki</span>
      </Button>
    </>
  );
  const wikiPartialStrip = wikiPartial ? (
    <div
      role="status"
      aria-label="Wiki partial details"
      className="rounded-md border border-warning/50 bg-warning/10 px-3 py-2 text-xs text-text"
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="font-medium text-warning">Wiki partial</span>
        <span>
          {wikiPartial.succeeded} / {wikiPartial.attempted} Sessions succeeded
        </span>
        <span className="text-text-muted">
          {wikiPartial.failed} failed · {wikiPartial.skipped} skipped
        </span>
        {wikiPartial.dominantErrorCount > 0 ? (
          <span className="text-text-muted">
            Dominant error: {wikiPartial.dominantErrorCode} ({wikiPartial.dominantErrorCount})
          </span>
        ) : null}
      </div>
      {wikiPartialErrorHint ? (
        <p className="mt-1 text-text-muted">
          <span className="font-medium text-warning">{wikiPartialErrorHint.title}</span>
          <span className="ml-1">{wikiPartialErrorHint.detail}</span>
        </p>
      ) : null}
      {wikiPartial.repairText ? <p className="mt-1 text-text-muted">{wikiPartial.repairText}</p> : null}
      {wikiPartial.errors.length > 0 ? (
        <ul className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-text-muted">
          {wikiPartial.errors.map((error, index) => (
            <li key={`${error.sessionId}-${error.errorCode}-${index}`}>
              {error.sessionId ? `${error.sessionId}: ` : ""}
              {error.errorCode}
              {error.validationError ? ` · ${error.validationError}` : ""}
              {error.strategy ? ` · ${error.strategy}` : ""}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  ) : undefined;
  const toolbarViewControls =
    viewMode === SESSION_LAYOUT_MODE ? (
      <SessionMapControls showCrossLinks={showSessionCrossLinks} onToggleCrossLinks={toggleSessionCrossLinks} />
    ) : viewMode === "wiki_map" ? (
      <WikiMapControls showCrossLinks={showWikiCrossLinks} onToggleCrossLinks={toggleWikiCrossLinks} onReset={resetLayout} />
    ) : null;
  const toolbarSearch = (
    <label className="relative block min-w-0">
      <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-faint" />
      <Input
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        className="pl-9"
        placeholder="Search graph nodes"
        aria-label="Search graph nodes"
      />
    </label>
  );
  const toolbarTypeFilters = (
    <div className="flex flex-wrap gap-2" aria-label="Node type filters">
      {nodeTypes.map((type) => {
        const active = enabledTypes.has(type);
        const colors = colorFor(type);
        return (
          <button
            key={type}
            type="button"
            aria-pressed={active}
            onClick={() => toggleType(type)}
            className={cn(
              "inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors focus-ring",
              active ? "border-border-strong bg-bg text-text" : "border-border bg-bg-subtle text-text-muted",
            )}
          >
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: colors.border }} />
            {typeLabel(type)}
          </button>
        );
      })}
    </div>
  );
  const showCanvasChrome = !loading && !loadError && !empty && !filteredEmpty;
  const overlayItems: GraphCanvasOverlayItem[] = [];
  if (activeJob) {
    overlayItems.push({
      id: "graph-build-progress",
      node: (
        <div
          role="status"
          aria-label="Graph build progress"
          className="rounded-md border border-border bg-surface/95 px-3 py-2 text-xs text-text shadow-sm"
        >
          <span className="font-medium">{status?.last_build_job?.stage ?? "queued"}</span>
          <span className="ml-2 text-text-muted">{formatPercent(status?.last_build_job?.progress)}</span>
        </div>
      ),
    });
  }
  if (activeWikiJob) {
    overlayItems.push({
      id: "wiki-refresh-progress",
      node: (
        <div
          role="status"
          aria-label="Wiki refresh progress"
          className="rounded-md border border-border bg-surface/95 px-3 py-2 text-xs text-text shadow-sm"
        >
          <span className="font-medium">Wiki · {wikiJob?.stage ?? "queued"}</span>
          <span className="ml-2 text-text-muted">{formatPercent(wikiJob?.progress)}</span>
        </div>
      ),
    });
  }
  if (viewMode === "wiki_map" && focusedWikiTopicId) {
    overlayItems.push({
      id: "wiki-focus",
      interactive: true,
      node: (
        <div role="status" aria-label="Wiki branch focus" className="flex items-center gap-2 rounded-md border border-accent/40 bg-surface/95 px-3 py-2 text-xs text-text shadow-sm">
          <span>{query.trim() ? "Branch focus is suspended while searching." : "Showing one Wiki branch."}</span>
          <Button variant="outline" size="sm" onClick={clearWikiFocus}>Return to full map</Button>
        </div>
      ),
    });
  }
  if (wikiJobErrorHint && !loading && !loadError) {
    overlayItems.push({
      id: "wiki-refresh-job-error",
      node: (
        <div role="status" aria-label="Wiki refresh failed" className="rounded-md border border-warning/50 bg-warning/10 px-3 py-2 text-xs text-text shadow-sm">
          <span className="block font-medium text-warning">Wiki refresh failed</span>
          <span className="mt-1 block text-text-muted">
            {wikiJobErrorCode}
            {wikiJobErrorCode ? " · " : ""}
            {wikiJobErrorHint.title}
          </span>
          <span className="mt-1 block text-text-muted">{wikiJobErrorHint.detail}</span>
        </div>
      ),
    });
  }
  if (showCanvasChrome) {
    overlayItems.push({
      id: "graph-canvas-controls",
      interactive: true,
      node: (
        <div aria-label="Graph canvas controls" className="flex flex-wrap gap-2">
          <Button variant="outline" size="sm" aria-pressed={viewportLocked} onClick={() => setViewportLocked((current) => !current)}>
            {viewportLocked ? <Lock className="h-4 w-4" /> : <Unlock className="h-4 w-4" />}
            <span className="max-[520px]:sr-only">{viewportLocked ? "Locked" : "Lock view"}</span>
          </Button>
          <Button variant="outline" size="sm" onClick={requestFitView}>
            <Maximize2 className="h-4 w-4" />
            <span className="max-[520px]:sr-only">Fit view</span>
          </Button>
          <Button variant="outline" size="sm" onClick={resetLayout}>
            <RotateCcw className="h-4 w-4" />
            <span className="max-[520px]:sr-only">Reset layout</span>
          </Button>
        </div>
      ),
    });
  }
  if (crossLinkNotice && showCanvasChrome) {
    overlayItems.push({
      id: "cross-link-notice",
      node: (
        <div role="status" className="rounded-md border border-border bg-surface/95 px-3 py-2 text-left text-xs text-text shadow-sm">
          <span className="block font-medium">{crossLinkNotice.title}</span>
          <span className="mt-1 block text-text-muted">{crossLinkNotice.detail}</span>
        </div>
      ),
    });
  }
  if (rebuildError && !loading && !loadError) {
    overlayItems.push({
      id: "graph-rebuild-error",
      node: (
        <div role="status" className="rounded-md border border-warning/50 bg-warning/10 px-3 py-2 text-xs text-text shadow-sm">
          <span className="block font-medium text-warning">Graph rebuild did not start</span>
          <span className="mt-1 block text-text-muted">{rebuildError}</span>
        </div>
      ),
    });
  }
  if (wikiRefreshError && !loading && !loadError) {
    overlayItems.push({
      id: "wiki-refresh-error",
      node: (
        <div role="status" className="rounded-md border border-warning/50 bg-warning/10 px-3 py-2 text-xs text-text shadow-sm">
          <span className="block font-medium text-warning">Wiki refresh did not start</span>
          <span className="mt-1 block text-text-muted">{wikiRefreshError}</span>
        </div>
      ),
    });
  }

  return (
    <div className="flex min-h-full flex-col bg-bg">
      <GraphPageHeader
        icon={<Workflow className="h-6 w-6 text-accent" aria-hidden />}
        title="Knowledge Graph"
        subtitle="Session-centered map across reports, notes, sources, and memory."
        badges={graphStatusBadges}
        actions={graphActions}
        statusStrip={wikiPartialStrip}
      />

      <div className="grid min-h-0 flex-1 grid-rows-[auto_minmax(520px,1fr)_auto] gap-0 lg:grid-cols-[minmax(0,1fr)_320px] lg:grid-rows-[auto_minmax(0,1fr)]">
        <div className="lg:col-span-2">
          <GraphToolbar
            tabs={<GraphViewTabs value={viewMode} onChange={changeViewMode} />}
            sessionFilter={
              <SessionFilter
                key={viewMode}
                sessions={sessionOptions}
                mode={activeSessionFilter.mode}
                selectedIds={effectiveSelectedSessionIds}
                onToggleSession={toggleSessionFilter}
                onShowAll={showAllSessions}
                onShowCrossLinked={showCrossLinkedSessions}
                onClear={clearSessionFilter}
                onResetDefault={resetSessionFilter}
              />
            }
            viewControls={toolbarViewControls}
            search={toolbarSearch}
            typeFilters={toolbarTypeFilters}
          />
        </div>

        <section className="relative min-h-[520px] overflow-hidden bg-bg-subtle" aria-label="Knowledge graph canvas">
          {loading ? (
            <div className="grid h-full min-h-[520px] place-items-center text-sm text-text-muted">
              <span className="inline-flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" /> Loading graph
              </span>
            </div>
          ) : loadError ? (
            <div className="grid h-full min-h-[520px] place-items-center px-6 text-center">
              <div>
                <ServerCrash className="mx-auto mb-3 h-8 w-8 text-danger" />
                <p className="text-sm font-medium text-text">Could not load graph</p>
                <p className="mt-1 text-sm text-text-muted">{loadError}</p>
              </div>
            </div>
          ) : empty ? (
            <div className="grid h-full min-h-[520px] place-items-center px-6 text-center">
              <div>
                <Workflow className="mx-auto mb-3 h-8 w-8 text-accent" />
                {viewMode === "wiki_map" ? (
                  <>
                    <p className="text-sm font-medium text-text">No Wiki Map yet</p>
                    <p className="mt-1 text-sm text-text-muted">Create a knowledge topic tree from the active Session Map.</p>
                    <Button className="mt-4" size="sm" onClick={() => void refreshWiki()} disabled={IS_DEMO || wikiRefreshing || activeWikiJob || !status?.active_version}>
                      {wikiRefreshing || activeWikiJob ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                      <span className="max-[520px]:sr-only">Create Wiki Map</span>
                    </Button>
                  </>
                ) : (
                  <>
                    <p className="text-sm font-medium text-text">No research graph yet</p>
                    <p className="mt-1 text-sm text-text-muted">Run a manual rebuild to project existing research assets.</p>
                    <Button className="mt-4" size="sm" onClick={() => void rebuild()} disabled={IS_DEMO || rebuilding}>
                      {rebuilding ? <Loader2 className="h-4 w-4 animate-spin" /> : <Workflow className="h-4 w-4" />}
                      <span className="max-[520px]:sr-only">Rebuild graph</span>
                    </Button>
                  </>
                )}
              </div>
            </div>
          ) : filteredEmpty ? (
            <div className="grid h-full min-h-[520px] place-items-center px-6 text-center">
              <div>
                <Workflow className="mx-auto mb-3 h-8 w-8 text-accent" />
                <p className="text-sm font-medium text-text">
                  {effectiveSelectedSessionIds.size === 0
                    ? manualSelectionMissing
                      ? "Selected Sessions are not in this graph"
                      : "No Sessions selected"
                    : "No graph nodes match the current filters"}
                </p>
                <p className="mt-1 text-sm text-text-muted">
                  {effectiveSelectedSessionIds.size === 0
                    ? manualSelectionMissing
                      ? "Restore the default or show all Sessions to bring the map back into view."
                      : "Choose Sessions or show all to bring the map back into view."
                    : "Adjust the Session, search, or type filters to see more of this map."}
                </p>
                {effectiveSelectedSessionIds.size === 0 ? (
                  <Button className="mt-4" size="sm" onClick={resetSessionFilter}>
                    Restore default Sessions
                  </Button>
                ) : null}
              </div>
            </div>
          ) : (
            <ReactFlowProvider>
              <GraphFlowCanvas
                nodes={displayNodes}
                edges={flowEdges}
                locked={viewportLocked}
                fitViewRequest={fitViewRequest}
                viewMode={currentLayoutMode}
                versionId={versionId}
                onNodesChange={handleNodesChange}
                onNodeClick={(node) => void selectNode(node)}
                onNodeMouseEnter={setHoveredNodeId}
                onNodeMouseLeave={() => setHoveredNodeId(undefined)}
                onEdgeClick={selectWikiCrossLink}
                onNodeDoubleClick={(node) => {
                  if (node.type !== "sourceHub" && node.type !== "memoryHub" && node.type !== "wikiTopic") {
                    navigate(resolveGraphJumpTarget(node));
                  }
                }}
              />
            </ReactFlowProvider>
          )}
          <GraphCanvasOverlayStack items={overlayItems} />
        </section>

        <aside className="border-t border-border bg-surface lg:border-l lg:border-t-0" aria-label="Node detail">
          <div className="space-y-4 p-5">
            <div>
              <h2 className="text-sm font-semibold text-text">Node detail</h2>
              <p className="mt-1 text-xs text-text-muted">
                Click a node for evidence. Double-click a node to return to its source surface.
              </p>
            </div>
            {selectedNode ? selectedNode.type === "wikiTopic" ? (
              <WikiInspector
                selectedNode={selectedNode}
                selectedTopic={selectedWikiTopic}
                selectedCrossLink={selectedWikiCrossLink}
                wikiMap={wikiMap}
                wikiPartial={wikiPartial}
                visibleTopicIds={visibleWikiTopicIds}
                onSelectTopic={selectWikiTopicById}
                onOpenSession={(sessionId) => navigate(`/session/${sessionId}`)}
                focusedTopicId={focusedWikiTopicId}
                onFocusTopic={focusWikiTopic}
                onClearFocus={clearWikiFocus}
                onOpenEvidence={navigate}
              />
            ) : (
              <div className="space-y-3">
                <div>
                  <div className="mb-2 flex flex-wrap items-center gap-2">
                    <Badge>{typeLabel(selectedNode.type)}</Badge>
                    <Badge variant="accent">{formatPercent(selectedNode.confidence)} confidence</Badge>
                    {detailLoading && <Loader2 className="h-4 w-4 animate-spin text-text-muted" aria-label="Loading node detail" />}
                  </div>
                  <h3 className="text-base font-semibold text-text">
                    {selectedNode.type === "memory" || selectedNode.type === "note" ? graphNodeLabel(selectedNode) : selectedNode.title}
                  </h3>
                  {detailSummary ? <p className="mt-2 text-sm text-text-muted">{detailSummary}</p> : null}
                </div>

                {reverseLinks.length > 0 ? (
                  <div>
                    <h4 className="mb-2 text-xs font-semibold uppercase text-text-faint">Sessions referencing this memory</h4>
                    <ul className="space-y-1.5">
                      {reverseLinks.slice(0, 8).map((item) => (
                        <li key={item.node_key} className="rounded-md border border-border bg-bg px-2 py-1.5 text-xs text-text-body">
                          <button type="button" className="text-left font-medium text-accent" onClick={() => navigate(item.jump_uri || `/session/${item.session_id}`)}>
                            {item.title || item.session_id}
                          </button>
                          {item.summary ? <span className="mt-1 block text-text-muted">{item.summary}</span> : null}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}

                {nodeAttrRows(selectedNode).length > 0 && (
                  <Card>
                    <CardContent className="space-y-2 py-3">
                      {nodeAttrRows(selectedNode).map(([key, value]) => (
                        <div key={key} className="min-w-0 text-xs">
                          <span className="block font-medium text-text-muted">{key}</span>
                          <span className="block break-words text-text-body">{value}</span>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                )}

                {"edges" in selectedNode && selectedNode.edges.length > 0 && (
                  <div>
                    <h4 className="mb-2 text-xs font-semibold uppercase text-text-faint">Relations</h4>
                    <ul className="space-y-1.5">
                      {selectedNode.edges.slice(0, 8).map((edge) => (
                        <li key={edge.id} className="rounded-md border border-border bg-bg px-2 py-1.5 text-xs text-text-body">
                          {edge.type.replaceAll("_", " ")} · {formatPercent(edge.confidence)}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {"provenance" in selectedNode && selectedNode.provenance.length > 0 && (
                  <div>
                    <h4 className="mb-2 text-xs font-semibold uppercase text-text-faint">Evidence</h4>
                    <ul className="space-y-1.5">
                      {selectedNode.provenance.slice(0, 6).map((item) => (
                        <li key={item.id} className="rounded-md border border-border bg-bg px-2 py-1.5 text-xs text-text-body">
                          <span className="font-medium">{item.source_kind}</span>
                          <span className="ml-1 text-text-muted">{item.source_id}</span>
                          {item.created_at ? <span className="ml-1 text-text-faint">{formatTime(item.created_at)}</span> : null}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {selectedNode.type === "concept" ? (
                  <p className="rounded-md border border-border bg-bg-subtle px-3 py-2 text-xs text-text-muted">
                    Concepts help discover canonical evidence. They are not citations or evidence sources themselves.
                  </p>
                ) : (
                  <Button variant="outline" size="sm" onClick={() => navigate(resolveGraphJumpTarget(selectedNode))}>
                    Open source surface
                  </Button>
                )}
              </div>
            ) : (
              <div className="rounded-md border border-dashed border-border p-4 text-sm text-text-muted">
                Select a node to inspect provenance and jump targets.
              </div>
            )}
          </div>
        </aside>

        <div
          className="flex flex-col gap-2 border-t border-border bg-surface px-5 py-3 text-xs lg:col-span-2 lg:flex-row lg:items-center lg:justify-between lg:px-6"
          aria-label="Graph status"
        >
          <p className="text-text-muted">
            Showing {shownCount} of {totalCount} nodes.{" "}
            {viewMode === "wiki_map" ? "Wiki Map uses the active wiki snapshot." : "Session Map is the default view."}
          </p>
          {!loading && !loadError && !empty && !filteredEmpty ? (
            <p className="max-w-full shrink-0 text-text" aria-label="Edge legend">
              <span className="font-medium">Solid</span>
              <span className="ml-1 text-text-muted">{viewMode === "wiki_map" ? "topic branch" : "projected relation"}</span>
              <span className="ml-3 font-medium text-rose-700">Dashed</span>
              <span className="ml-1 text-text-muted">
                {viewMode === "wiki_map"
                  ? showWikiCrossLinks
                    ? "cross-session evidence"
                    : "cross-session evidence hidden"
                  : showSessionCrossLinks
                    ? "memory + cross-session"
                    : "memory reference; cross-session hidden"}
              </span>
            </p>
          ) : null}
        </div>
      </div>
    </div>
  );
}
