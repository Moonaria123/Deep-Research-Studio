import { describe, expect, it, vi, beforeEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import type { ReactNode } from "react";
import { Graph } from "./Graph";
import { api, ApiError } from "@/lib/api";
import type { KnowledgeGraphGraph, KnowledgeGraphSessionMap, KnowledgeGraphStatus, KnowledgeGraphWikiMap } from "@/lib/types";

const reactFlowMock = vi.hoisted(() => ({
  fitView: vi.fn(),
  setViewport: vi.fn(),
  getViewport: vi.fn(() => ({ x: 0, y: 0, zoom: 1 })),
  lastProps: undefined as any,
}));

function createLocalStorageMock(): Storage {
  let values = new Map<string, string>();
  return {
    get length() {
      return values.size;
    },
    clear: vi.fn(() => {
      values = new Map<string, string>();
    }),
    getItem: vi.fn((key: string) => values.get(key) ?? null),
    key: vi.fn((index: number) => Array.from(values.keys())[index] ?? null),
    removeItem: vi.fn((key: string) => {
      values.delete(key);
    }),
    setItem: vi.fn((key: string, value: string) => {
      values.set(key, String(value));
    }),
  };
}

vi.mock("@xyflow/react", () => ({
  applyNodeChanges: (changes: Array<any>, nodes: Array<any>) =>
    nodes.map((node) => {
      const positionChange = changes.find((change) => change.id === node.id && change.type === "position");
      return positionChange?.position ? { ...node, position: positionChange.position } : node;
    }),
  useReactFlow: () => ({
    fitView: reactFlowMock.fitView,
    setViewport: reactFlowMock.setViewport,
    getViewport: reactFlowMock.getViewport,
  }),
  ReactFlowProvider: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  ReactFlow: ({
    nodes,
    edges,
    nodeTypes,
    onNodesChange,
    onNodeClick,
    onNodeDoubleClick,
    onNodeMouseEnter,
    onNodeMouseLeave,
    children,
    ...props
  }: {
    nodes: Array<{ id: string; type: string; position: { x: number; y: number }; data: { label: string; graphNode: unknown } }>;
    edges: unknown[];
    nodeTypes?: Record<string, (props: any) => ReactNode>;
    onNodesChange?: (changes: Array<any>) => void;
    onNodeClick?: (event: unknown, node: { id: string; data: { graphNode: unknown } }) => void;
    onNodeDoubleClick?: (event: unknown, node: { id: string; data: { graphNode: unknown } }) => void;
    onNodeMouseEnter?: (event: unknown, node: { id: string }) => void;
    onNodeMouseLeave?: () => void;
    children: ReactNode;
  }) => {
    reactFlowMock.lastProps = { nodes, edges, nodeTypes, onNodesChange, ...props };
    return (
      <div aria-label="mock graph canvas">
        {nodes.map((node) => {
          const NodeComponent = nodeTypes?.[node.type];
          return (
            <button
              key={node.id}
              type="button"
              aria-label={node.data.label}
              data-node-id={node.id}
              data-x={node.position.x}
              data-y={node.position.y}
              onClick={(event) => onNodeClick?.(event, node)}
              onDoubleClick={(event) => onNodeDoubleClick?.(event, node)}
              onMouseEnter={(event) => onNodeMouseEnter?.(event, node)}
              onMouseLeave={() => onNodeMouseLeave?.()}
            >
              {NodeComponent ? <NodeComponent id={node.id} type={node.type} data={node.data} selected={false} /> : node.data.label}
            </button>
          );
        })}
        <span>{edges.length} graph edges</span>
        {children}
      </div>
    );
  },
  Background: () => <div data-testid="graph-background" />,
  Controls: () => <div data-testid="graph-controls" />,
  MiniMap: () => <div data-testid="graph-minimap" />,
  Handle: () => null,
  Position: {
    Left: "left",
    Right: "right",
    Top: "top",
    Bottom: "bottom",
  },
  MarkerType: { ArrowClosed: "arrowclosed" },
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      getKnowledgeGraphStatus: vi.fn(),
      getKnowledgeGraph: vi.fn(),
      getKnowledgeGraphSessionMap: vi.fn(),
      getKnowledgeGraphWikiMap: vi.fn(),
      refreshKnowledgeGraphWikiMap: vi.fn(),
      getKnowledgeGraphWikiMapJob: vi.fn(),
      getKnowledgeGraphNode: vi.fn(),
      rebuildKnowledgeGraph: vi.fn(),
    },
  };
});

const statusReady: KnowledgeGraphStatus = {
  workspace_id: "default",
  active_version: {
    id: "kgv-1",
    workspace_id: "default",
    status: "active",
    is_active: true,
  },
  last_build_job: {
    id: "kgjob-1",
    workspace_id: "default",
    backend: "thread",
    status: "succeeded",
    stage: "completed",
    progress: 1,
  },
  counts: { nodes: 2, edges: 1 },
};

const graphReady: KnowledgeGraphGraph = {
  workspace_id: "default",
  active_version: statusReady.active_version,
  nodes: [
    {
      id: "node-session",
      type: "session",
      key: "session:sess-1",
      title: "AI safety",
      summary: "Assistant summary for the latest graph refresh.",
      confidence: 1,
      visible: true,
      attrs: {
        session_id: "sess-1",
        jump_uri: "/session/sess-1",
        assistant_summary: "Assistant summary for the latest graph refresh.",
      },
    },
    {
      id: "node-source",
      type: "source",
      key: "source:doc-1",
      title: "Safety source",
      summary: "example.com",
      confidence: 1,
      visible: true,
      attrs: { document_id: "doc-1", url: "https://example.com/safety", jump_uri: "/library/sources/doc-1" },
    },
  ],
  edges: [
    {
      id: "edge-1",
      type: "session_used_source",
      source: "node-session",
      target: "node-source",
      weight: 1,
      confidence: 1,
      visible: true,
      attrs: {},
    },
  ],
};

const graphWithConcept: KnowledgeGraphGraph = {
  ...graphReady,
  nodes: [
    ...graphReady.nodes,
    {
      id: "node-concept",
      type: "concept",
      key: "concept:ai-safety",
      title: "AI safety concept",
      summary: "Deterministic concept projection.",
      confidence: 0.9,
      visible: true,
      attrs: {
        normalized_label: "ai safety",
        evidence_ids: ["source:doc-1"],
      },
    },
  ],
  edges: [
    ...graphReady.edges,
    {
      id: "edge-concept-source",
      type: "concept_supported_by_source",
      source: "node-concept",
      target: "node-source",
      weight: 1,
      confidence: 0.9,
      visible: true,
      attrs: {},
    },
  ],
};

const sessionMapReady: KnowledgeGraphSessionMap = {
  workspace_id: "default",
  active_version: statusReady.active_version,
  layout_mode: "session_tree",
  session_limit: 50,
  member_preview: 100,
  sessions: [
    {
      session: {
        ...graphReady.nodes[0],
        assistant_summary: "Assistant summary for the latest graph refresh.",
        assistant_excerpt: "Assistant summary for the latest graph refresh.",
        last_assistant_message_id: "msg-1",
        last_assistant_at: "123",
      },
      reports: [],
      notes: [],
      source_set: { count: 1, node_keys: ["source:doc-1"], members: [graphReady.nodes[1]] },
      memory_set: {
        count: 1,
        node_keys: ["memory:mem-1"],
        members: [
          {
            id: "node-memory",
            type: "memory",
            key: "memory:mem-1",
            title: "Memory mem-1",
            summary: "Shared memory summary",
            confidence: 1,
            visible: true,
            attrs: { namespace: "workspace" },
            referencing_sessions: [
              {
                node_key: "session:sess-1",
                session_id: "sess-1",
                title: "AI safety",
                summary: "Assistant summary for the latest graph refresh.",
                jump_uri: "/session/sess-1",
              },
            ],
          },
        ],
      },
      counts: { reports: 0, notes: 0, sources: 1, memories: 1 },
    },
  ],
  nodes_by_key: {
    "session:sess-1": graphReady.nodes[0],
    "source:doc-1": graphReady.nodes[1],
    "memory:mem-1": {
      id: "node-memory",
      type: "memory",
      key: "memory:mem-1",
      title: "Memory mem-1",
      summary: "Shared memory summary",
      confidence: 1,
      visible: true,
      attrs: { namespace: "workspace" },
      referencing_sessions: [
        {
          node_key: "session:sess-1",
          session_id: "sess-1",
          title: "AI safety",
          summary: "Assistant summary for the latest graph refresh.",
          jump_uri: "/session/sess-1",
        },
      ],
    },
  },
  edges: [],
};

function sessionNode(id: string, title: string) {
  return {
    id: `node-${id}`,
    type: "session" as const,
    key: `session:${id}`,
    title,
    summary: `${title} summary`,
    confidence: 1,
    visible: true,
    attrs: {
      session_id: id,
      jump_uri: `/session/${id}`,
      assistant_summary: `${title} summary`,
    },
    assistant_summary: `${title} summary`,
    assistant_excerpt: `${title} summary`,
    last_assistant_message_id: `msg-${id}`,
    last_assistant_at: "123",
  };
}

function sessionLane(id: string, title: string): KnowledgeGraphSessionMap["sessions"][number] {
  return {
    session: sessionNode(id, title),
    reports: [],
    notes: [],
    source_set: { count: 0, node_keys: [], members: [] },
    memory_set: { count: 0, node_keys: [], members: [] },
    counts: { reports: 0, notes: 0, sources: 0, memories: 0 },
  };
}

function multiSessionMap(count = 10): KnowledgeGraphSessionMap {
  const sessions = Array.from({ length: count }, (_, index) => sessionLane(`sess-${index + 1}`, `Session ${index + 1}`));
  return {
    ...sessionMapReady,
    sessions,
    nodes_by_key: Object.fromEntries(sessions.map((lane) => [lane.session.key, lane.session])),
  };
}

function sharedMemorySessionMap(): KnowledgeGraphSessionMap {
  const sharedMemory = {
    id: "node-memory-shared",
    type: "memory" as const,
    key: "memory:mem-shared",
    title: "Memory mem-shared",
    summary: "Cross-session memory summary",
    confidence: 1,
    visible: true,
    attrs: { namespace: "workspace" },
    referencing_sessions: [
      {
        node_key: "session:sess-a",
        session_id: "sess-a",
        title: "Session A",
        summary: "Session A summary",
        jump_uri: "/session/sess-a",
      },
      {
        node_key: "session:sess-b",
        session_id: "sess-b",
        title: "Session B",
        summary: "Session B summary",
        jump_uri: "/session/sess-b",
      },
    ],
  };
  const sessions = ["a", "b"].map((suffix) => {
    const id = `sess-${suffix}`;
    return {
      ...sessionLane(id, `Session ${suffix.toUpperCase()}`),
      memory_set: { count: 1, node_keys: [sharedMemory.key], members: [sharedMemory] },
      counts: { reports: 0, notes: 0, sources: 0, memories: 1 },
    };
  });
  return {
    ...sessionMapReady,
    sessions,
    nodes_by_key: {
      ...Object.fromEntries(sessions.map((lane) => [lane.session.key, lane.session])),
      [sharedMemory.key]: sharedMemory,
    },
  };
}

function sharedSourceSessionMap(): KnowledgeGraphSessionMap {
  const sharedSource = {
    id: "node-source-shared",
    type: "source" as const,
    key: "source:shared",
    title: "Shared source",
    summary: "shared.example",
    confidence: 1,
    visible: true,
    attrs: { document_id: "doc-shared", url: "https://shared.example/research" },
  };
  const sessions = ["1", "2", "3"].map((suffix) => {
    const id = `sess-${suffix}`;
    const sourceKeys = suffix === "3" ? [`source:solo-${suffix}`] : [sharedSource.key];
    const sources =
      suffix === "3"
        ? [
            {
              ...sharedSource,
              id: "node-source-solo-3",
              key: "source:solo-3",
              title: "Solo source",
            },
          ]
        : [sharedSource];
    return {
      ...sessionLane(id, `Session ${suffix}`),
      source_set: { count: 1, node_keys: sourceKeys, members: sources },
      counts: { reports: 0, notes: 0, sources: 1, memories: 0 },
    };
  });
  return {
    ...sessionMapReady,
    sessions,
    nodes_by_key: {
      ...Object.fromEntries(sessions.map((lane) => [lane.session.key, lane.session])),
      [sharedSource.key]: sharedSource,
      "source:solo-3": {
        ...sharedSource,
        id: "node-source-solo-3",
        key: "source:solo-3",
        title: "Solo source",
      },
    },
  };
}

const wikiMapReady: KnowledgeGraphWikiMap = {
  workspace_id: "default",
  layout_mode: "wiki_map",
  active_graph_version: statusReady.active_version,
  active_snapshot: {
    id: "kgwiki-1",
    workspace_id: "default",
    graph_version_id: "kgv-1",
    status: "succeeded",
    is_active: true,
    stale: false,
    stale_reason: "",
    stats: { topics: 3, tree_edges: 2, cross_links: 1 },
  },
  stale: false,
  stale_reason: "",
  stats: { topics: 3, tree_edges: 2, cross_links: 1 },
  topics: [
    {
      id: "topic-root",
      snapshot_id: "kgwiki-1",
      key: "wiki:root",
      title: "Workspace Wiki",
      summary: "Knowledge map synthesized from completed research sessions.",
      depth: 0,
      parent_id: "",
      session_ids: [],
      source_message_ids: [],
      attrs: { kind: "workspace_root" },
    },
    {
      id: "topic-session",
      snapshot_id: "kgwiki-1",
      key: "wiki:session:sess-1",
      title: "AI safety",
      summary: "Session topic root.",
      depth: 1,
      parent_id: "topic-root",
      session_ids: ["sess-1"],
      source_message_ids: ["msg-1"],
      attrs: { kind: "session_root", session_id: "sess-1", session_title: "AI safety" },
    },
    {
      id: "topic-risk",
      snapshot_id: "kgwiki-1",
      key: "wiki:topic:sess-1:taxonomy",
      title: "Risk taxonomy",
      summary: "Group model behavior and operational risks.",
      depth: 2,
      parent_id: "topic-session",
      session_ids: ["sess-1"],
      source_message_ids: ["msg-1"],
      attrs: {
        kind: "topic",
        session_id: "sess-1",
        session_title: "AI safety",
        schema_version: "2",
        concept_key: "risk-taxonomy",
        confidence: "high",
        quality: "single-evidence",
        evidence_refs: [{ type: "source", id: "source:doc-1", excerpt: "Safety source excerpt." }],
      },
    },
  ],
  tree_edges: [
    { id: "wiki-edge-1", snapshot_id: "kgwiki-1", source: "topic-root", target: "topic-session", type: "parent_child", attrs: {} },
    { id: "wiki-edge-2", snapshot_id: "kgwiki-1", source: "topic-session", target: "topic-risk", type: "parent_child", attrs: {} },
  ],
  cross_links: [
    {
      id: "wiki-cross-1",
      snapshot_id: "kgwiki-1",
      source: "topic-session",
      target: "topic-risk",
      type: "shared_kg_evidence",
      evidence_type: "shared_kg_evidence",
      evidence_refs: [{ node_key: "source:doc-1", type: "source" }],
      attrs: {
        confidence: "high",
        reason: "Both sessions discuss Risk taxonomy and share 1 evidence item.",
      },
    },
  ],
  last_refresh_job: {
    id: "kgwikijob-1",
    snapshot_id: "kgwiki-1",
    graph_version_id: "kgv-1",
    workspace_id: "default",
    backend: "thread",
    backend_job_id: "kgwikijob-1",
    status: "succeeded",
    stage: "completed",
    progress: 1,
    attempted_sessions: 1,
    succeeded_sessions: 1,
    skipped_sessions: 0,
    failed_sessions: 0,
    stats: { topics: 3 },
  },
};

function wikiMapWithCrossSessionLink(): KnowledgeGraphWikiMap {
  return {
    ...wikiMapReady,
    stats: { topics: 3, tree_edges: 1, cross_links: 1 },
    topics: [
      {
        id: "topic-sess-1",
        snapshot_id: "kgwiki-1",
        key: "wiki:sess-1",
        title: "Session 1 topic",
        summary: "",
        depth: 1,
        parent_id: "",
        session_ids: ["sess-1"],
        source_message_ids: [],
        attrs: { kind: "session_root", session_id: "sess-1" },
      },
      {
        id: "topic-sess-2",
        snapshot_id: "kgwiki-1",
        key: "wiki:sess-2",
        title: "Session 2 topic",
        summary: "",
        depth: 1,
        parent_id: "",
        session_ids: ["sess-2"],
        source_message_ids: [],
        attrs: { kind: "session_root", session_id: "sess-2" },
      },
      {
        id: "topic-sess-3",
        snapshot_id: "kgwiki-1",
        key: "wiki:sess-3",
        title: "Session 3 linked topic",
        summary: "",
        depth: 2,
        parent_id: "topic-sess-2",
        session_ids: ["sess-3"],
        source_message_ids: [],
        attrs: { kind: "topic", session_id: "sess-3" },
      },
    ],
    tree_edges: [
      { id: "tree-23", snapshot_id: "kgwiki-1", source: "topic-sess-2", target: "topic-sess-3", type: "parent_child", attrs: {} },
    ],
    cross_links: [
      {
        id: "cross-23",
        snapshot_id: "kgwiki-1",
        source: "topic-sess-2",
        target: "topic-sess-3",
        type: "shared_kg_evidence",
        evidence_type: "shared_kg_evidence",
        evidence_refs: [],
        attrs: { confidence: "high", reason: "Both sessions discuss a linked topic and share 1 evidence item." },
      },
    ],
  };
}

function wikiMapWithoutCrossLinks(overrides: Partial<KnowledgeGraphWikiMap> = {}): KnowledgeGraphWikiMap {
  return {
    ...wikiMapReady,
    stats: { topics: wikiMapReady.topics.length, tree_edges: wikiMapReady.tree_edges.length, cross_links: 0 },
    cross_links: [],
    active_snapshot: wikiMapReady.active_snapshot
      ? {
          ...wikiMapReady.active_snapshot,
          stale: false,
          stale_reason: "",
          stats: { topics: wikiMapReady.topics.length, tree_edges: wikiMapReady.tree_edges.length, cross_links: 0 },
        }
      : null,
    stale: false,
    stale_reason: "",
    ...overrides,
  };
}

function wikiMapPartialWithErrors(): KnowledgeGraphWikiMap {
  const errors = Array.from({ length: 12 }, (_, index) => ({
    session_id: index === 0 ? "sess-failed" : `sess-failed-${index + 1}`,
    error_code: "INVALID_EVIDENCE_REF",
    validation_error: "INVALID_EVIDENCE_REF",
    structured_strategy: "json_schema_strict",
    request_id: index === 0 ? "req-hidden" : undefined,
    raw_transcript: index === 0 ? "RAW_TRANSCRIPT_SHOULD_NOT_RENDER" : undefined,
  }));
  const stats = {
    topics: 3,
    tree_edges: 2,
    cross_links: 0,
    attempted_sessions: 15,
    succeeded_sessions: 6,
    failed_sessions: 9,
    skipped_sessions: 18,
    evidence_ref_dropped: 2,
    topic_dropped: 1,
    errors,
  };
  return {
    ...wikiMapReady,
    active_snapshot: wikiMapReady.active_snapshot
      ? {
          ...wikiMapReady.active_snapshot,
          status: "partial",
          stats,
        }
      : null,
    stats,
    last_refresh_job: wikiMapReady.last_refresh_job
      ? {
          ...wikiMapReady.last_refresh_job,
          status: "partial",
          stage: "partial",
          attempted_sessions: 15,
          succeeded_sessions: 6,
          failed_sessions: 9,
          skipped_sessions: 18,
          stats,
        }
      : null,
  };
}

function wikiMapWithFailedProviderJob(): KnowledgeGraphWikiMap {
  return {
    ...wikiMapReady,
    last_refresh_job: wikiMapReady.last_refresh_job
      ? {
          ...wikiMapReady.last_refresh_job,
          id: "kgwikijob-provider-failed",
          status: "failed",
          stage: "failed",
          attempted_sessions: 0,
          succeeded_sessions: 0,
          failed_sessions: 0,
          skipped_sessions: 0,
          error_code: "PROVIDER_NOT_CONFIGURED",
          error_message: "RAW_PROVIDER_ERROR_SHOULD_NOT_RENDER",
          stats: {
            error_code: "PROVIDER_NOT_CONFIGURED",
            request_id: "req-hidden",
            raw_prompt: "RAW_PROMPT_SHOULD_NOT_RENDER",
          },
        }
      : null,
  };
}

function wikiMapWithDeepBranch(): KnowledgeGraphWikiMap {
  return {
    ...wikiMapReady,
    stats: { topics: 6, tree_edges: 5, cross_links: 1 },
    topics: [
      ...wikiMapReady.topics,
      {
        id: "topic-risk-detail",
        snapshot_id: "kgwiki-1",
        key: "wiki:topic:sess-1:taxonomy:detail",
        title: "Operational risk detail",
        summary: "A third-level topic hidden until the branch is expanded.",
        depth: 3,
        parent_id: "topic-risk",
        session_ids: ["sess-1"],
        source_message_ids: ["msg-1"],
        attrs: { kind: "topic", session_id: "sess-1", session_title: "AI safety" },
      },
      {
        id: "topic-session-2",
        snapshot_id: "kgwiki-1",
        key: "wiki:session:sess-2",
        title: "Climate policy",
        summary: "A second Session root that should remain visible while expanding another branch.",
        depth: 1,
        parent_id: "topic-root",
        session_ids: ["sess-2"],
        source_message_ids: ["msg-2"],
        attrs: { kind: "session_root", session_id: "sess-2", session_title: "Climate policy" },
      },
      {
        id: "topic-policy",
        snapshot_id: "kgwiki-1",
        key: "wiki:topic:sess-2:policy",
        title: "Policy evidence",
        summary: "A second Session topic that should remain in the forest.",
        depth: 2,
        parent_id: "topic-session-2",
        session_ids: ["sess-2"],
        source_message_ids: ["msg-2"],
        attrs: { kind: "topic", session_id: "sess-2", session_title: "Climate policy" },
      },
    ],
    tree_edges: [
      ...wikiMapReady.tree_edges,
      { id: "wiki-edge-3", snapshot_id: "kgwiki-1", source: "topic-risk", target: "topic-risk-detail", type: "parent_child", attrs: {} },
      { id: "wiki-edge-4", snapshot_id: "kgwiki-1", source: "topic-root", target: "topic-session-2", type: "parent_child", attrs: {} },
      { id: "wiki-edge-5", snapshot_id: "kgwiki-1", source: "topic-session-2", target: "topic-policy", type: "parent_child", attrs: {} },
    ],
  };
}

function wikiMapWithHiddenPeer(): KnowledgeGraphWikiMap {
  return {
    ...wikiMapReady,
    stats: { topics: 5, tree_edges: 3, cross_links: 1 },
    topics: [
      ...wikiMapReady.topics,
      {
        id: "topic-session-peer",
        snapshot_id: "kgwiki-1",
        key: "wiki:session:sess-peer",
        title: "Peer research",
        summary: "A related peer Session root.",
        depth: 1,
        parent_id: "topic-root",
        session_ids: ["sess-peer"],
        source_message_ids: ["msg-peer"],
        attrs: { kind: "session_root", session_id: "sess-peer", session_title: "Peer research" },
      },
      {
        id: "topic-peer-policy",
        snapshot_id: "kgwiki-1",
        key: "wiki:topic:sess-peer:policy",
        title: "Climate policy",
        summary: "A peer topic hidden by the current search query.",
        depth: 2,
        parent_id: "topic-session-peer",
        session_ids: ["sess-peer"],
        source_message_ids: ["msg-peer"],
        attrs: {
          kind: "topic",
          session_id: "sess-peer",
          session_title: "Peer research",
          concept_key: "risk-taxonomy",
          confidence: "high",
          quality: "grounded",
          evidence_refs: [{ type: "message", id: "msg-peer", excerpt: "Peer evidence." }],
        },
      },
    ],
    tree_edges: [
      ...wikiMapReady.tree_edges,
      { id: "wiki-edge-peer-root", snapshot_id: "kgwiki-1", source: "topic-root", target: "topic-session-peer", type: "parent_child", attrs: {} },
      { id: "wiki-edge-peer-topic", snapshot_id: "kgwiki-1", source: "topic-session-peer", target: "topic-peer-policy", type: "parent_child", attrs: {} },
    ],
    cross_links: [
      {
        id: "wiki-cross-peer",
        snapshot_id: "kgwiki-1",
        source: "topic-risk",
        target: "topic-peer-policy",
        type: "shared_kg_evidence",
        evidence_type: "shared_kg_evidence",
        evidence_refs: [{ node_key: "source:doc-1", type: "source" }],
        attrs: {
          confidence: "high",
          reason: "Both topics share source evidence.",
        },
      },
    ],
  };
}

function renderGraph(initialPath = "/graph") {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <Routes>
        <Route path="/graph" element={<Graph />} />
        <Route path="/library/sources/:documentId" element={<div>Library source route</div>} />
        <Route path="/library/reports/:reportId" element={<div>Library report route</div>} />
        <Route path="/library/notes/:noteId" element={<div>Library note route</div>} />
        <Route path="/library/reports" element={<div>Library reports tab</div>} />
        <Route path="/library/notes" element={<div>Library notes tab</div>} />
        <Route path="/library" element={<div>Library route</div>} />
        <Route path="/session/:sessionId" element={<div>Session route</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("Graph", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: createLocalStorageMock(),
    });
    window.localStorage.clear();
    reactFlowMock.lastProps = undefined;
    reactFlowMock.getViewport.mockReturnValue({ x: 0, y: 0, zoom: 1 });
    vi.mocked(api.getKnowledgeGraphStatus).mockResolvedValue(statusReady);
    vi.mocked(api.getKnowledgeGraph).mockResolvedValue(graphReady);
    vi.mocked(api.getKnowledgeGraphSessionMap).mockResolvedValue(sessionMapReady);
    vi.mocked(api.getKnowledgeGraphWikiMap).mockResolvedValue(wikiMapReady);
    vi.mocked(api.refreshKnowledgeGraphWikiMap).mockResolvedValue({
      ...wikiMapReady.last_refresh_job!,
      id: "kgwikijob-2",
      status: "queued",
      stage: "queued",
      progress: 0,
    });
    vi.mocked(api.getKnowledgeGraphWikiMapJob).mockResolvedValue({
      ...wikiMapReady.last_refresh_job!,
      id: "kgwikijob-2",
      status: "succeeded",
      stage: "completed",
      progress: 1,
    });
    vi.mocked(api.getKnowledgeGraphNode).mockImplementation(async (nodeId: string) => {
      const baseNode = nodeId === "node-memory" ? sessionMapReady.nodes_by_key["memory:mem-1"] : graphReady.nodes[0];
      return {
        node: {
          ...baseNode,
          edges: graphReady.edges,
          provenance: [
            {
              id: "prov-1",
              owner_kind: "node",
              source_kind: "session",
              source_id: "sess-1",
              span: {},
              evidence: {},
            },
          ],
        },
      };
    });
  });

  it("renders graph nodes and opens node detail with provenance", async () => {
    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });
    expect(screen.getByText("2 nodes · 1 edges")).toBeInTheDocument();
    expect(screen.getByTestId("graph-minimap")).toBeInTheDocument();
    expect(screen.getByText("Assistant summary for the latest graph refresh.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Session Map" })).toHaveAttribute("aria-pressed", "true");

    fireEvent.click(screen.getByRole("button", { name: "AI safety" }));

    await waitFor(() => expect(api.getKnowledgeGraphNode).toHaveBeenCalledWith("node-session"));
    expect(await screen.findByText("Evidence")).toBeInTheDocument();
    expect(screen.getByText(/session used source/i)).toBeInTheDocument();
  });

  it("keeps concepts opt-in and explains that they are discovery aids rather than citations", async () => {
    vi.mocked(api.getKnowledgeGraph).mockResolvedValue(graphWithConcept);
    vi.mocked(api.getKnowledgeGraphNode).mockImplementation(async (nodeId: string) => {
      const baseNode = graphWithConcept.nodes.find((node) => node.id === nodeId) ?? graphWithConcept.nodes[0];
      return {
        node: {
          ...baseNode,
          edges: graphWithConcept.edges.filter((edge) => edge.source === nodeId || edge.target === nodeId),
          provenance: [],
        },
      };
    });
    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });
    expect(screen.getByRole("button", { name: "Concept" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.queryByRole("button", { name: "AI safety concept" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Sources 1" }));
    fireEvent.click(screen.getByRole("button", { name: "Concept" }));
    fireEvent.click(await screen.findByRole("button", { name: "AI safety concept" }));

    await waitFor(() => expect(api.getKnowledgeGraphNode).toHaveBeenCalledWith("node-concept"));
    expect(screen.getByText(/not citations or evidence sources themselves/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /open source surface/i })).not.toBeInTheDocument();
  });

  it("shows Wiki partial diagnostics without rendering raw failure payloads", async () => {
    vi.mocked(api.getKnowledgeGraphWikiMap).mockResolvedValue(wikiMapPartialWithErrors());
    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });

    expect(screen.getAllByText("Wiki partial")).toHaveLength(2);
    expect(screen.getByLabelText("Wiki partial details")).toHaveTextContent("6 / 15 Sessions succeeded");
    expect(screen.getByText("Dominant error: INVALID_EVIDENCE_REF (12)")).toBeInTheDocument();
    expect(screen.getByText("Evidence 引用不合法")).toBeInTheDocument();
    expect(screen.getByText("这是旧快照常见错误；请在当前版本重新 Refresh Wiki，让系统用新的 evidence allowlist 生成。")).toBeInTheDocument();
    expect(screen.getByText("2 dropped refs · 1 dropped topics")).toBeInTheDocument();
    expect(screen.getByText("sess-failed: INVALID_EVIDENCE_REF · INVALID_EVIDENCE_REF · json_schema_strict")).toBeInTheDocument();
    expect(screen.queryByText("req-hidden")).not.toBeInTheDocument();
    expect(screen.queryByText("RAW_TRANSCRIPT_SHOULD_NOT_RENDER")).not.toBeInTheDocument();
  });

  it("shows actionable failed Wiki job hints without rendering raw provider payloads", async () => {
    vi.mocked(api.getKnowledgeGraphWikiMap).mockResolvedValue(wikiMapWithFailedProviderJob());
    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });

    expect(screen.getByLabelText("Wiki refresh failed")).toHaveTextContent("PROVIDER_NOT_CONFIGURED");
    expect(screen.getByLabelText("Wiki refresh failed")).toHaveTextContent("Memory 模型未配置");
    expect(screen.getByText("到 Settings 的 Memory 角色绑定可用模型，运行 Test memory format 后再 Refresh Wiki。")).toBeInTheDocument();
    expect(screen.queryByText("RAW_PROVIDER_ERROR_SHOULD_NOT_RENDER")).not.toBeInTheDocument();
    expect(screen.queryByText("req-hidden")).not.toBeInTheDocument();
    expect(screen.queryByText("RAW_PROMPT_SHOULD_NOT_RENDER")).not.toBeInTheDocument();
  });

  it("renders responsive Graph chrome regions without React Flow default controls", async () => {
    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });

    expect(screen.getByLabelText("Graph page header")).toBeInTheDocument();
    expect(screen.getByLabelText("Graph toolbar")).toHaveAttribute("data-layout", "graph-toolbar-grid");
    expect(screen.getByLabelText("Graph canvas overlay stack")).toContainElement(screen.getByLabelText("Graph canvas controls"));
    expect(screen.queryByTestId("graph-controls")).not.toBeInTheDocument();
  });

  it("stacks graph jobs, controls, diagnostics, and enqueue errors in the canvas overlay safe area", async () => {
    vi.mocked(api.getKnowledgeGraphWikiMap).mockResolvedValue({
      ...wikiMapReady,
      last_refresh_job: {
        ...wikiMapReady.last_refresh_job!,
        status: "running",
        stage: "distilling",
        progress: 0.5,
      },
    });
    vi.mocked(api.rebuildKnowledgeGraph).mockRejectedValue(
      new ApiError("KNOWLEDGE_GRAPH_BUILD_RUNNING", "knowledge graph rebuild is already running"),
    );
    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });
    fireEvent.click(screen.getByRole("button", { name: /rebuild graph/i }));

    const stack = screen.getByLabelText("Graph canvas overlay stack");
    expect(stack).toContainElement(screen.getByLabelText("Wiki refresh progress"));
    expect(stack).toContainElement(screen.getByLabelText("Graph canvas controls"));
    expect(stack).toContainElement(await screen.findByText("Graph rebuild did not start"));
    expect(screen.getByLabelText("Wiki refresh progress")).not.toHaveClass("absolute");
    expect(screen.getByLabelText("Graph canvas controls")).not.toHaveClass("absolute");
  });

  it("defaults to the first eight Sessions and can show all", async () => {
    vi.mocked(api.getKnowledgeGraphSessionMap).mockResolvedValue(multiSessionMap(10));
    renderGraph();

    await screen.findByRole("button", { name: "Session 1" });
    expect(reactFlowMock.lastProps.nodes.filter((node: any) => node.type === "sessionCard")).toHaveLength(8);
    expect(reactFlowMock.lastProps.nodes.some((node: any) => node.data.graphNode.title === "Session 9")).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: /Sessions · 8 \/ 10/ }));
    fireEvent.click(screen.getByRole("button", { name: "Show all" }));

    expect(reactFlowMock.lastProps.nodes.filter((node: any) => node.type === "sessionCard")).toHaveLength(10);
    expect(reactFlowMock.lastProps.nodes.some((node: any) => node.data.graphNode.title === "Session 9")).toBe(true);
  });

  it("keeps manual Session selection independent across Session Map and Wiki Map", async () => {
    vi.mocked(api.getKnowledgeGraphSessionMap).mockResolvedValue(multiSessionMap(3));
    vi.mocked(api.getKnowledgeGraphWikiMap).mockResolvedValue(wikiMapWithCrossSessionLink());
    renderGraph();

    await screen.findByRole("button", { name: "Session 1" });
    fireEvent.click(screen.getByRole("button", { name: /Sessions · 3 \/ 3/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Clear" }));
    expect(await screen.findByText("No Sessions selected")).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Session 2"));

    expect(reactFlowMock.lastProps.nodes.filter((node: any) => node.type === "sessionCard")).toHaveLength(1);
    expect(reactFlowMock.lastProps.nodes[0].data.graphNode.title).toBe("Session 2");

    fireEvent.click(screen.getByRole("button", { name: "Wiki Map" }));

    expect(screen.getByRole("button", { name: /Sessions · 3 \/ 3/ })).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Session 1 topic" })).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Session 2 topic" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Session 3 linked topic" })).toBeInTheDocument();
    expect(reactFlowMock.lastProps.edges.map((edge: any) => edge.id)).not.toContain("wiki-cross-cross-23");

    fireEvent.click(screen.getByRole("button", { name: /Sessions · 3 \/ 3/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Clear" }));
    fireEvent.click(screen.getByLabelText("Session 3"));
    expect(screen.queryByRole("button", { name: "Session 1 topic" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Session 2 topic" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Session 3 linked topic" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Session Map" }));

    expect(await screen.findByRole("button", { name: "Session 2" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Session 1" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Wiki Map" }));
    fireEvent.click(screen.getByRole("button", { name: "Show cross-session links" }));

    const crossEdge = reactFlowMock.lastProps.edges.find((edge: any) => edge.id === "wiki-cross-cross-23");
    expect(crossEdge.style.strokeDasharray).toBe("6 4");
    expect(crossEdge.style.opacity).toBeLessThanOrEqual(0.5);
  });

  it("keeps node positions stable and does not refit on hover", async () => {
    renderGraph();

    const session = await screen.findByRole("button", { name: "AI safety" });
    await waitFor(() => expect(reactFlowMock.fitView).toHaveBeenCalledTimes(1));
    const before = reactFlowMock.lastProps.nodes.find((node: any) => node.data.graphNode.id === "node-session").position;

    fireEvent.mouseEnter(session);
    fireEvent.mouseLeave(session);

    const after = reactFlowMock.lastProps.nodes.find((node: any) => node.data.graphNode.id === "node-session").position;
    expect(after).toEqual(before);
    expect(reactFlowMock.fitView).toHaveBeenCalledTimes(1);
  });

  it("expands Source and Memory hubs in the default Session Map", async () => {
    window.localStorage.setItem("kg-viewport:default:session_tree:kgv-1", JSON.stringify({ x: 44, y: 55, zoom: 0.9 }));
    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });
    await waitFor(() => expect(reactFlowMock.fitView).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(reactFlowMock.setViewport).toHaveBeenCalledTimes(1));
    expect(screen.queryByRole("button", { name: "Safety source" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Sources 1" }));
    expect(screen.getByRole("button", { name: "Safety source" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Memories 1" }));
    fireEvent.click(screen.getByRole("button", { name: "Shared memory summary" }));
    await act(async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 0));
    });

    await screen.findByText("Sessions referencing this memory");
    expect(screen.getAllByText("AI safety").length).toBeGreaterThanOrEqual(2);
    expect(reactFlowMock.fitView).toHaveBeenCalledTimes(1);
    expect(reactFlowMock.setViewport).toHaveBeenCalledTimes(1);
  });

  it("toggles Session Map cross-session memory links independently from Wiki Map links", async () => {
    vi.mocked(api.getKnowledgeGraphSessionMap).mockResolvedValue(sharedMemorySessionMap());
    renderGraph();

    await screen.findByRole("button", { name: "Session A" });
    screen.getAllByRole("button", { name: "Memories 1" }).forEach((button) => fireEvent.click(button));
    expect(reactFlowMock.lastProps.edges.map((edge: any) => edge.id).filter((id: string) => id.startsWith("session-cross-"))).toEqual([]);

    fireEvent.click(screen.getByRole("button", { name: "Show cross-session links" }));

    const crossEdges = reactFlowMock.lastProps.edges.filter((edge: any) => edge.id.startsWith("session-cross-"));
    expect(crossEdges.map((edge: any) => edge.id)).toEqual([
      "session-cross-memory:mem-shared-sess-a-sess-b",
      "session-cross-memory:mem-shared-sess-b-sess-a",
    ]);
    expect(crossEdges[0].style.strokeDasharray).toBe("6 4");
    expect(crossEdges[0].style.opacity).toBeLessThanOrEqual(0.5);
    expect(window.localStorage.getItem("kg-session-cross-links:default:kgv-1")).toBe("true");

    fireEvent.click(screen.getByRole("button", { name: "Hide cross-session links" }));

    expect(reactFlowMock.lastProps.edges.map((edge: any) => edge.id).filter((id: string) => id.startsWith("session-cross-"))).toEqual([]);
    expect(window.localStorage.getItem("kg-session-cross-links:default:kgv-1")).toBe("false");

    fireEvent.click(screen.getByRole("button", { name: "Wiki Map" }));

    expect(await screen.findByRole("button", { name: "Show cross-session links" })).toBeInTheDocument();
    expect(window.localStorage.getItem("kg-wiki-cross-links:default:kgwiki-1")).toBeNull();
  });

  it("shows a Session Map no-link diagnostic when cross-links are enabled without visible edges", async () => {
    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });
    fireEvent.click(screen.getByRole("button", { name: "Show cross-session links" }));

    expect(screen.getByText("No cross-session links available for the current Session filter.")).toBeInTheDocument();
    expect(screen.getByText("Shared source or memory links require matching visible Sessions and expanded members.")).toBeInTheDocument();
    expect(reactFlowMock.lastProps.edges.map((edge: any) => edge.id).filter((id: string) => id.startsWith("session-cross-"))).toEqual([]);
  });

  it("shows Wiki Map no-link and stale diagnostics when cross-links are enabled without visible edges", async () => {
    vi.mocked(api.getKnowledgeGraphWikiMap).mockResolvedValue(wikiMapWithoutCrossLinks({ stale: true, stale_reason: "graph_version_changed" }));
    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });
    fireEvent.click(screen.getByRole("button", { name: "Wiki Map" }));
    fireEvent.click(await screen.findByRole("button", { name: "Show cross-session links" }));

    expect(screen.getByText("No Wiki cross-session links in this snapshot.")).toBeInTheDocument();
    expect(screen.getByText("Snapshot is stale: graph_version_changed. Refresh Wiki to rebuild evidence links.")).toBeInTheDocument();
    expect(reactFlowMock.lastProps.edges.map((edge: any) => edge.id).filter((id: string) => id.startsWith("wiki-cross-"))).toEqual([]);
  });

  it("renders the Wiki Inspector and explains hidden peer topics without changing filters", async () => {
    vi.mocked(api.getKnowledgeGraphWikiMap).mockResolvedValue(wikiMapWithHiddenPeer());
    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });
    fireEvent.click(screen.getByRole("button", { name: "Wiki Map" }));
    fireEvent.change(screen.getByLabelText("Search graph nodes"), { target: { value: "Risk taxonomy" } });
    fireEvent.click(await screen.findByRole("button", { name: "Risk taxonomy" }));

    expect(screen.getByLabelText("Wiki Inspector")).toHaveTextContent("Risk taxonomy");
    expect(screen.getByText("Concept key:")).toBeInTheDocument();
    expect(screen.getByText("Safety source excerpt.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Climate policy" }));

    expect(screen.getByText("Peer topic is hidden by the current view.")).toBeInTheDocument();
    expect(screen.getByText("Climate policy is not visible under the current Session, search, or type filters.")).toBeInTheDocument();
    expect(screen.getByLabelText("Search graph nodes")).toHaveValue("Risk taxonomy");
  });

  it("renders visible Session Map edges and separates Source and Memory member columns", async () => {
    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });
    fireEvent.click(screen.getByRole("button", { name: "Sources 1" }));
    fireEvent.click(screen.getByRole("button", { name: "Memories 1" }));

    const sourceNode = reactFlowMock.lastProps.nodes.find((node: any) => node.id === "source:session:sess-1:source:doc-1");
    const memoryNode = reactFlowMock.lastProps.nodes.find((node: any) => node.id === "memory:session:sess-1:memory:mem-1");
    expect(sourceNode.position.x).toBe(1080);
    expect(memoryNode.position.x).toBe(1560);

    const solidEdge = reactFlowMock.lastProps.edges.find((edge: any) => edge.id === "session-hub-session:sess-1-source");
    const dashedEdge = reactFlowMock.lastProps.edges.find((edge: any) => edge.id === "memory-member-session:sess-1-memory:mem-1");
    expect(solidEdge.markerEnd).toEqual({ type: "arrowclosed", color: "#475569", width: 18, height: 18 });
    expect(solidEdge.style.opacity).toBeGreaterThanOrEqual(0.7);
    expect(dashedEdge.markerEnd).toEqual({ type: "arrowclosed", color: "#e11d48", width: 18, height: 18 });
    expect(dashedEdge.style.strokeDasharray).toBe("6 4");
    expect(dashedEdge.style.opacity).toBeGreaterThanOrEqual(0.8);
  });

  it("expands Memory member cards without using raw memory ids as the main title", async () => {
    vi.mocked(api.getKnowledgeGraphSessionMap).mockResolvedValue({
      ...sessionMapReady,
      sessions: [
        {
          ...sessionMapReady.sessions[0],
          memory_set: {
            ...sessionMapReady.sessions[0].memory_set,
            members: [
              {
                ...sessionMapReady.nodes_by_key["memory:mem-1"],
                summary: "",
                attrs: { namespace: "workspace", memory_id: "mem-1", excerpt: "Memory excerpt from a completed session." },
              },
            ],
          },
        },
      ],
      nodes_by_key: {
        ...sessionMapReady.nodes_by_key,
        "memory:mem-1": {
          ...sessionMapReady.nodes_by_key["memory:mem-1"],
          summary: "",
          attrs: { namespace: "workspace", memory_id: "mem-1", excerpt: "Memory excerpt from a completed session." },
        },
      },
    });
    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });
    fireEvent.click(screen.getByRole("button", { name: "Memories 1" }));
    const memoryTitle = screen.getByTestId("graph-node-title-node-memory");
    expect(memoryTitle).toHaveTextContent("No memory summary yet.");
    expect(memoryTitle).not.toHaveTextContent("Memory mem-1");

    fireEvent.click(screen.getByTestId("graph-node-node-memory").closest("button") as HTMLButtonElement);
    expect(await screen.findByText("Open memory")).toBeInTheDocument();
    expect(screen.getByText("Memory excerpt from a completed session.")).toBeInTheDocument();
  });

  it("expands Note member cards without using raw note ids as the main title", async () => {
    const noteNode = {
      id: "node-note",
      type: "note" as const,
      key: "note:note-1",
      title: "Note note-1",
      summary: "Post-run note summary",
      confidence: 1,
      visible: true,
      attrs: {
        note_id: "note-1",
        namespace: "note",
        content_excerpt: "Note excerpt from a completed session.",
        jump_uri: "/library/notes/note-1",
      },
    };
    vi.mocked(api.getKnowledgeGraphSessionMap).mockResolvedValue({
      ...sessionMapReady,
      sessions: [
        {
          ...sessionMapReady.sessions[0],
          notes: [noteNode],
          counts: { ...sessionMapReady.sessions[0].counts, notes: 1 },
        },
      ],
      nodes_by_key: {
        ...sessionMapReady.nodes_by_key,
        "note:note-1": noteNode,
      },
    });
    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });
    const noteTitle = screen.getByTestId("graph-node-title-node-note");
    expect(noteTitle).toHaveTextContent("Post-run note summary");
    expect(noteTitle).not.toHaveTextContent("Note note-1");

    fireEvent.click(screen.getByTestId("graph-node-node-note").closest("button") as HTMLButtonElement);
    expect(await screen.findByText("Open note")).toBeInTheDocument();
    expect(screen.getByText("Note excerpt from a completed session.")).toBeInTheDocument();
  });

  it("shows note placeholder title when summary is missing", async () => {
    const noteNode = {
      id: "node-note",
      type: "note" as const,
      key: "note:note-1",
      title: "Note note-1",
      summary: "",
      confidence: 1,
      visible: true,
      attrs: { note_id: "note-1", namespace: "note", content_excerpt: "Fallback excerpt text." },
    };
    vi.mocked(api.getKnowledgeGraphSessionMap).mockResolvedValue({
      ...sessionMapReady,
      sessions: [{ ...sessionMapReady.sessions[0], notes: [noteNode], counts: { ...sessionMapReady.sessions[0].counts, notes: 1 } }],
      nodes_by_key: { ...sessionMapReady.nodes_by_key, "note:note-1": noteNode },
    });
    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });
    expect(screen.getByTestId("graph-node-title-node-note")).toHaveTextContent("No note summary yet.");
  });

  it("toggles Session cards between collapsed and expanded states", async () => {
    renderGraph();

    const session = await screen.findByRole("button", { name: "AI safety" });
    expect(screen.getByText("Collapsed")).toBeInTheDocument();
    expect(screen.queryByText("Message msg-1")).not.toBeInTheDocument();

    fireEvent.click(session);

    expect(await screen.findByText("Expanded")).toBeInTheDocument();
    expect(screen.getByText("Message msg-1")).toBeInTheDocument();

    fireEvent.click(session);

    expect(await screen.findByText("Collapsed")).toBeInTheDocument();
    expect(screen.queryByText("Message msg-1")).not.toBeInTheDocument();
  });

  it("keeps Session Map viewport storage isolated under the retained tab shell", async () => {
    renderGraph();
    await screen.findByRole("button", { name: "AI safety" });
    expect(screen.queryByRole("button", { name: "ER View" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Wiki Map" })).toBeInTheDocument();

    act(() => {
      reactFlowMock.getViewport.mockReturnValue({ x: 11, y: 22, zoom: 1.1 });
      reactFlowMock.lastProps.onMoveEnd();
    });

    expect(window.localStorage.getItem("kg-viewport:default:session_tree:kgv-1")).toBe(
      JSON.stringify({ x: 11, y: 22, zoom: 1.1 }),
    );
    expect(window.localStorage.getItem("kg-viewport:default:wiki_map:kgwiki-1")).toBeNull();
  });

  it("renders the Wiki Map topic tree and keeps viewport storage separate", async () => {
    renderGraph();
    await screen.findByRole("button", { name: "AI safety" });

    fireEvent.click(screen.getByRole("button", { name: "Wiki Map" }));

    expect(await screen.findByRole("button", { name: "Risk taxonomy" })).toBeInTheDocument();
    expect(screen.getByText("1 source")).toBeInTheDocument();
    expect(screen.getByText("single-evidence")).toBeInTheDocument();
    expect(screen.getByText("3 wiki topics")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show cross-session links" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Hide cross-session links" })).not.toBeInTheDocument();
    expect(reactFlowMock.lastProps.edges.map((edge: any) => edge.id)).not.toContain("wiki-cross-wiki-cross-1");
    const topicNode = reactFlowMock.lastProps.nodes.find((node: any) => node.id === "topic-risk");
    const rootNode = reactFlowMock.lastProps.nodes.find((node: any) => node.id === "topic-session");
    expect(topicNode.position.x).toBeGreaterThan(rootNode.position.x);
    fireEvent.click(screen.getByRole("button", { name: "Show cross-session links" }));
    const crossEdge = reactFlowMock.lastProps.edges.find((edge: any) => edge.id === "wiki-cross-wiki-cross-1");
    expect(crossEdge.style.strokeDasharray).toBe("6 4");
    expect(screen.getByRole("button", { name: "Hide cross-session links" })).toBeInTheDocument();
    expect(window.localStorage.getItem("kg-wiki-cross-links:default:kgwiki-1")).toBe("true");

    fireEvent.click(screen.getByRole("button", { name: "Risk taxonomy" }));
    expect(await screen.findByText("Topic evidence")).toBeInTheDocument();
    expect(screen.getAllByText("source:doc-1").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Safety source excerpt.")).toBeInTheDocument();
    expect(screen.getByText("Both sessions discuss Risk taxonomy and share 1 evidence item.")).toBeInTheDocument();

    act(() => {
      reactFlowMock.getViewport.mockReturnValue({ x: 33, y: 44, zoom: 0.8 });
      reactFlowMock.lastProps.onMoveEnd();
    });

    expect(window.localStorage.getItem("kg-viewport:default:wiki_map:kgwiki-1")).toBe(
      JSON.stringify({ x: 33, y: 44, zoom: 0.8 }),
    );
    expect(window.localStorage.getItem("kg-viewport:default:session_tree:kgv-1")).toBeNull();
  });

  it("uses view-specific With links semantics for Session Map and Wiki Map", async () => {
    vi.mocked(api.getKnowledgeGraphSessionMap).mockResolvedValue(sharedSourceSessionMap());
    vi.mocked(api.getKnowledgeGraphWikiMap).mockResolvedValue(wikiMapWithCrossSessionLink());
    renderGraph();
    await screen.findByRole("button", { name: "Session 1" });

    fireEvent.click(screen.getByRole("button", { name: "Sessions · 3 / 3" }));
    fireEvent.click(await screen.findByRole("button", { name: "With links" }));
    expect(screen.getByRole("button", { name: "Sessions · 2 / 3" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Session 1" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Session 2" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Session 3" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Wiki Map" }));

    expect(await screen.findByRole("button", { name: "Session 1 topic" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sessions · 3 / 3" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Sessions · 3 / 3" }));
    fireEvent.click(await screen.findByRole("button", { name: "With links" }));

    expect(screen.getByRole("button", { name: "Sessions · 2 / 3" })).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Session 2 topic" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Session 3 linked topic" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Session 1 topic" })).not.toBeInTheDocument();
  });

  it("collapses deep Wiki topics by default and expands them in place without focusing the branch", async () => {
    const lanes = [sessionLane("sess-1", "AI safety"), sessionLane("sess-2", "Climate policy")];
    vi.mocked(api.getKnowledgeGraphSessionMap).mockResolvedValue({
      ...sessionMapReady,
      sessions: lanes,
      nodes_by_key: Object.fromEntries(lanes.map((lane) => [lane.session.key, lane.session])),
    });
    vi.mocked(api.getKnowledgeGraphWikiMap).mockResolvedValue(wikiMapWithDeepBranch());
    renderGraph();
    await screen.findByRole("button", { name: "AI safety" });

    fireEvent.click(screen.getByRole("button", { name: "Wiki Map" }));

    expect(await screen.findByRole("button", { name: "Risk taxonomy" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Policy evidence" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Operational risk detail" })).not.toBeInTheDocument();
    expect(screen.getAllByText("+1").length).toBeGreaterThanOrEqual(1);

    fireEvent.click(screen.getByRole("button", { name: "Risk taxonomy" }));

    expect(await screen.findByRole("button", { name: "Operational risk detail" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Policy evidence" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Return full map" })).not.toBeInTheDocument();
    expect(window.localStorage.getItem("kg-wiki-focus:default:kgwiki-1")).toBeNull();
  });

  it("focuses an explicit Wiki branch, suspends it during search, and restores it without losing manual expansion", async () => {
    const lanes = [sessionLane("sess-1", "AI safety"), sessionLane("sess-2", "Climate policy")];
    vi.mocked(api.getKnowledgeGraphSessionMap).mockResolvedValue({
      ...sessionMapReady,
      sessions: lanes,
      nodes_by_key: Object.fromEntries(lanes.map((lane) => [lane.session.key, lane.session])),
    });
    vi.mocked(api.getKnowledgeGraphWikiMap).mockResolvedValue(wikiMapWithDeepBranch());
    renderGraph();
    await screen.findByRole("button", { name: "AI safety" });
    fireEvent.click(screen.getByRole("button", { name: "Wiki Map" }));

    fireEvent.click(await screen.findByRole("button", { name: "Risk taxonomy" }));
    fireEvent.click(await screen.findByRole("button", { name: "Operational risk detail" }));
    fireEvent.click(screen.getByRole("button", { name: "Focus branch" }));

    expect(window.localStorage.getItem("kg-wiki-focus:default:kgwiki-1")).toBe("topic-risk");
    expect(screen.getByLabelText("Wiki branch focus")).toHaveTextContent("Showing one Wiki branch.");
    expect(screen.queryByRole("button", { name: "Policy evidence" })).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Search graph nodes"), { target: { value: "Policy evidence" } });
    expect(await screen.findByRole("button", { name: "Policy evidence" })).toBeInTheDocument();
    expect(screen.getByLabelText("Wiki branch focus")).toHaveTextContent("Branch focus is suspended while searching.");

    fireEvent.change(screen.getByLabelText("Search graph nodes"), { target: { value: "" } });
    expect(await screen.findByRole("button", { name: "Operational risk detail" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Policy evidence" })).not.toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("button", { name: "Return to full map" })[0]);
    expect(await screen.findByRole("button", { name: "Policy evidence" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Operational risk detail" })).toBeInTheDocument();
    expect(window.localStorage.getItem("kg-wiki-focus:default:kgwiki-1")).toBeNull();
  });

  it("opens Wiki cross-link edge details without changing link visibility or focus", async () => {
    renderGraph();
    await screen.findByRole("button", { name: "AI safety" });
    fireEvent.click(screen.getByRole("button", { name: "Wiki Map" }));
    fireEvent.click(screen.getByRole("button", { name: "Show cross-session links" }));

    const edge = reactFlowMock.lastProps.edges.find((item: any) => item.id === "wiki-cross-wiki-cross-1");
    act(() => reactFlowMock.lastProps.onEdgeClick({}, edge));

    expect(screen.getByLabelText("Cross-link detail")).toHaveTextContent("AI safety");
    expect(screen.getByLabelText("Cross-link detail")).toHaveTextContent("Risk taxonomy");
    expect(screen.getByLabelText("Cross-link detail")).toHaveTextContent("Both sessions discuss Risk taxonomy");
    expect(screen.getByRole("button", { name: "Hide cross-session links" })).toBeInTheDocument();
    expect(window.localStorage.getItem("kg-wiki-focus:default:kgwiki-1")).toBeNull();
  });

  it("starts Wiki Map refresh only from the explicit CTA", async () => {
    renderGraph();
    await screen.findByRole("button", { name: "AI safety" });

    expect(api.refreshKnowledgeGraphWikiMap).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Refresh Wiki" }));

    await waitFor(() => expect(api.refreshKnowledgeGraphWikiMap).toHaveBeenCalledWith({ workspace_id: "default" }));
  });

  it("filters Session Map member nodes without reintroducing ER view", async () => {
    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });
    fireEvent.click(screen.getByRole("button", { name: "Sources 1" }));
    fireEvent.change(screen.getByLabelText("Search graph nodes"), { target: { value: "source" } });
    expect(screen.getByRole("button", { name: "Safety source" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "ER View" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Source" }));
    expect(screen.queryByRole("button", { name: "Safety source" })).not.toBeInTheDocument();
  });

  it("renders bounded custom nodes for long source URLs", async () => {
    const longSource = {
      id: "node-source-long",
      type: "source" as const,
      key: "source:long",
      title: "https://example.com/a/very/long/path/that/should/not/become/the/main/visible/title",
      summary: "example.com",
      confidence: 1,
      visible: true,
      attrs: {
        document_id: "doc-long",
        url: "https://example.com/a/very/long/path/that/should/not/become/the/main/visible/title",
      },
    };
    vi.mocked(api.getKnowledgeGraph).mockResolvedValue({
      ...graphReady,
      nodes: [
        {
          ...longSource,
        },
      ],
      edges: [],
    });
    vi.mocked(api.getKnowledgeGraphSessionMap).mockResolvedValue({
      ...sessionMapReady,
      sessions: [
        {
          ...sessionMapReady.sessions[0],
          source_set: { count: 1, node_keys: ["source:long"], members: [longSource] },
          counts: { ...sessionMapReady.sessions[0].counts, sources: 1 },
        },
      ],
      nodes_by_key: {
        ...sessionMapReady.nodes_by_key,
        "source:long": longSource,
      },
    });
    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });
    fireEvent.click(screen.getByRole("button", { name: "Sources 1" }));
    await screen.findByRole("button", { name: /https:\/\/example.com/ });
    expect(screen.getByTestId("graph-node-node-source-long")).toHaveClass("w-[248px]");
    expect(screen.getByTestId("graph-node-title-node-source-long")).toHaveStyle({ overflowWrap: "anywhere" });
    expect(screen.getAllByText("example.com").length).toBeGreaterThan(0);
  });

  it("maps the viewport lock to React Flow interaction props", async () => {
    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });
    const statusBar = screen.getByLabelText("Graph status");
    expect(statusBar).toContainElement(screen.getByText("Solid"));
    expect(screen.getByText("Solid").parentElement).not.toHaveClass("absolute");
    expect(screen.getByText("Solid").parentElement).not.toHaveClass("top-4");
    expect(reactFlowMock.lastProps.nodesDraggable).toBe(true);

    fireEvent.click(screen.getByRole("button", { name: "Lock view" }));

    expect(screen.getByRole("button", { name: "Locked" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByLabelText("Graph canvas overlay stack")).toContainElement(screen.getByRole("button", { name: "Locked" }));
    expect(reactFlowMock.lastProps.nodesDraggable).toBe(false);
    expect(reactFlowMock.lastProps.nodesConnectable).toBe(false);
    expect(reactFlowMock.lastProps.panOnDrag).toBe(false);
    expect(reactFlowMock.lastProps.zoomOnScroll).toBe(false);
    expect(reactFlowMock.lastProps.zoomOnPinch).toBe(false);
    expect(reactFlowMock.lastProps.zoomOnDoubleClick).toBe(false);
    expect(reactFlowMock.lastProps.autoPanOnNodeDrag).toBe(false);
  });

  it("persists dragged node positions and resets the layout", async () => {
    const { unmount } = renderGraph();
    await screen.findByRole("button", { name: "AI safety" });

    act(() => {
      reactFlowMock.lastProps.onNodesChange([
        { id: "sessionCard:session:sess-1", type: "position", position: { x: 123, y: 456 }, dragging: false },
      ]);
    });

    expect(window.localStorage.getItem("kg-layout:default:session_tree:kgv-1:session:sess-1")).toBe(
      JSON.stringify({ x: 123, y: 456 }),
    );
    unmount();
    renderGraph();
    await screen.findByRole("button", { name: "AI safety" });
    expect(reactFlowMock.lastProps.nodes.find((node: any) => node.id === "sessionCard:session:sess-1").position).toEqual({
      x: 123,
      y: 456,
    });

    fireEvent.click(screen.getByRole("button", { name: "Reset layout" }));

    expect(window.localStorage.getItem("kg-layout:default:session_tree:kgv-1:session:sess-1")).toBeNull();
    expect(reactFlowMock.lastProps.nodes.find((node: any) => node.id === "sessionCard:session:sess-1").position).not.toEqual({
      x: 123,
      y: 456,
    });
  });

  it("does not persist drag changes while locked", async () => {
    renderGraph();
    await screen.findByRole("button", { name: "AI safety" });

    fireEvent.click(screen.getByRole("button", { name: "Lock view" }));
    act(() => {
      reactFlowMock.lastProps.onNodesChange([
        { id: "node-session", type: "position", position: { x: 222, y: 333 }, dragging: false },
      ]);
    });

    expect(window.localStorage.getItem("kg-layout:default:session_tree:kgv-1:session:sess-1")).toBeNull();
  });

  it("starts a manual rebuild from the empty state", async () => {
    vi.mocked(api.getKnowledgeGraphStatus).mockResolvedValue({
      workspace_id: "default",
      active_version: null,
      last_build_job: null,
      counts: { nodes: 0, edges: 0 },
    });
    vi.mocked(api.getKnowledgeGraph).mockResolvedValue({
      workspace_id: "default",
      active_version: null,
      nodes: [],
      edges: [],
    });
    vi.mocked(api.getKnowledgeGraphSessionMap).mockResolvedValue({
      workspace_id: "default",
      active_version: null,
      layout_mode: "session_tree",
      session_limit: 0,
      member_preview: 0,
      sessions: [],
      nodes_by_key: {},
      edges: [],
    });
    vi.mocked(api.getKnowledgeGraphWikiMap).mockResolvedValue({
      ...wikiMapReady,
      active_graph_version: null,
      active_snapshot: null,
      stale: false,
      stale_reason: "",
      stats: { topics: 0, tree_edges: 0, cross_links: 0 },
      topics: [],
      tree_edges: [],
      cross_links: [],
      last_refresh_job: null,
    });
    vi.mocked(api.rebuildKnowledgeGraph).mockResolvedValue({
      id: "kgjob-2",
      workspace_id: "default",
      backend: "thread",
      status: "queued",
      stage: "queued",
      progress: 0,
    });

    renderGraph();

    await screen.findByText("No research graph yet");
    const rebuildButtons = screen.getAllByRole("button", { name: /rebuild graph/i });
    fireEvent.click(rebuildButtons[rebuildButtons.length - 1]);

    await waitFor(() =>
      expect(api.rebuildKnowledgeGraph).toHaveBeenCalledWith({ workspace_id: "default" }),
    );
  });

  it("keeps rebuild 409 separate from graph load errors", async () => {
    vi.mocked(api.rebuildKnowledgeGraph).mockRejectedValue(
      new ApiError("KNOWLEDGE_GRAPH_BUILD_RUNNING", "knowledge graph rebuild is already running"),
    );

    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });
    fireEvent.click(screen.getByRole("button", { name: /rebuild graph/i }));

    expect(await screen.findByText("Graph rebuild did not start")).toBeInTheDocument();
    expect(screen.getByText("A graph rebuild is already running. This page will keep checking for updates.")).toBeInTheDocument();
    expect(screen.queryByText("Could not load graph")).not.toBeInTheDocument();
    await waitFor(() => expect(api.getKnowledgeGraphStatus).toHaveBeenCalledTimes(2));
  });

  it("shows load errors only when the graph read fails", async () => {
    vi.mocked(api.getKnowledgeGraph).mockRejectedValueOnce(new ApiError("INTERNAL_ERROR", "graph read failed"));

    renderGraph();

    expect(await screen.findByText("Could not load graph")).toBeInTheDocument();
    expect(screen.getByText("graph read failed")).toBeInTheDocument();
  });

  it("double-clicks source nodes to the Library sources deep link", async () => {
    renderGraph();

    await screen.findByRole("button", { name: "AI safety" });
    fireEvent.click(screen.getByRole("button", { name: "Sources 1" }));
    await screen.findByRole("button", { name: "Safety source" });
    fireEvent.doubleClick(screen.getByRole("button", { name: "Safety source" }));

    expect(await screen.findByText("Library source route")).toBeInTheDocument();
  });
});
