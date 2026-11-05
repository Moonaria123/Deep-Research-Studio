import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import type { ReactNode } from "react";
import { Graph } from "./Graph";
import { api } from "@/lib/api";
import type {
  KnowledgeGraphGraph,
  KnowledgeGraphSessionMap,
  KnowledgeGraphStatus,
  KnowledgeGraphWikiMap,
  KnowledgeGraphWikiRefreshJob,
} from "@/lib/types";

vi.mock("@xyflow/react", () => ({
  applyNodeChanges: (_changes: unknown[], nodes: unknown[]) => nodes,
  useReactFlow: () => ({
    fitView: vi.fn(),
    setViewport: vi.fn(),
    getViewport: vi.fn(() => ({ x: 0, y: 0, zoom: 1 })),
  }),
  ReactFlowProvider: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  ReactFlow: ({ children }: { children: ReactNode }) => (
    <div aria-label="mock graph canvas">{children}</div>
  ),
  Background: () => null,
  Controls: () => null,
  MiniMap: () => null,
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
      ...actual.api,
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

const activeVersion = {
  id: "kgv-t58",
  workspace_id: "default",
  status: "active",
  is_active: true,
};

const graphReady = {
  workspace_id: "default",
  active_version: activeVersion,
  nodes: [],
  edges: [],
} as KnowledgeGraphGraph;

const sessionMapReady = {
  workspace_id: "default",
  active_version: activeVersion,
  layout_mode: "session_tree",
  session_limit: 50,
  member_preview: 100,
  sessions: [],
  nodes_by_key: {},
  edges: [],
} as KnowledgeGraphSessionMap;

function buildJob(status: string, stage: string) {
  return {
    id: "kgjob-t58",
    workspace_id: "default",
    backend: "rq",
    backend_job_id: "kgjob-t58",
    status,
    stage,
    progress: status === "queued" || status === "running" ? 0.5 : 1,
  };
}

function wikiJob(status: string, stage: string): KnowledgeGraphWikiRefreshJob {
  return {
    id: "kgwikijob-t58",
    snapshot_id: "kgwiki-t58",
    workspace_id: "default",
    graph_version_id: activeVersion.id,
    backend: "rq",
    backend_job_id: "kgwikijob-t58",
    status,
    stage,
    progress: status === "queued" || status === "running" ? 0.5 : 1,
    attempted_sessions: 2,
    succeeded_sessions: status === "partial" ? 1 : 2,
    skipped_sessions: 0,
    failed_sessions: status === "partial" || status === "failed" ? 1 : 0,
    stats: {},
  };
}

function statusWithBuild(status: string, stage: string): KnowledgeGraphStatus {
  return {
    workspace_id: "default",
    active_version: activeVersion,
    last_build_job: buildJob(status, stage),
    counts: { nodes: 0, edges: 0 },
  } as KnowledgeGraphStatus;
}

function wikiMapWithJob(status: string, stage: string): KnowledgeGraphWikiMap {
  return {
    workspace_id: "default",
    layout_mode: "wiki_map",
    active_graph_version: activeVersion,
    active_snapshot: null,
    stale: true,
    stale_reason: "no_active_wiki_snapshot",
    stats: { topics: 0, tree_edges: 0, cross_links: 0 },
    topics: [],
    tree_edges: [],
    cross_links: [],
    last_refresh_job: wikiJob(status, stage),
  };
}

function idleWikiMap(): KnowledgeGraphWikiMap {
  return {
    ...wikiMapWithJob("succeeded", "completed"),
    last_refresh_job: null,
  } as KnowledgeGraphWikiMap;
}

function createLocalStorageMock(): Storage {
  const values = new Map<string, string>();
  return {
    get length() {
      return values.size;
    },
    clear: vi.fn(() => values.clear()),
    getItem: vi.fn((key: string) => values.get(key) ?? null),
    key: vi.fn((index: number) => Array.from(values.keys())[index] ?? null),
    removeItem: vi.fn((key: string) => values.delete(key)),
    setItem: vi.fn((key: string, value: string) => values.set(key, String(value))),
  };
}

function renderGraph() {
  return render(
    <MemoryRouter initialEntries={["/graph"]}>
      <Routes>
        <Route path="/graph" element={<Graph />} />
      </Routes>
    </MemoryRouter>,
  );
}

function sseFrame(seq: number, event: string, data: Record<string, unknown>): string {
  return `id: ${seq}\nevent: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function sseResponse(frames: string[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      frames.forEach((frame) => controller.enqueue(encoder.encode(frame)));
      controller.close();
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function requestUrl(input: RequestInfo | URL): string {
  return input instanceof Request ? input.url : String(input);
}

function usedLegacyPolling(intervalSpy: {
  mock: { calls: ReadonlyArray<ReadonlyArray<unknown>> };
}): boolean {
  return intervalSpy.mock.calls.some((call) => call[1] === 3000 || call[1] === 5000);
}

function installBaseApiMocks({
  status,
  wikiMap,
}: {
  status: KnowledgeGraphStatus;
  wikiMap: KnowledgeGraphWikiMap;
}) {
  vi.mocked(api.getKnowledgeGraphStatus).mockResolvedValue(status);
  vi.mocked(api.getKnowledgeGraph).mockResolvedValue(graphReady);
  vi.mocked(api.getKnowledgeGraphSessionMap).mockResolvedValue(sessionMapReady);
  vi.mocked(api.getKnowledgeGraphWikiMap).mockResolvedValue(wikiMap);
  vi.mocked(api.getKnowledgeGraphWikiMapJob).mockResolvedValue(
    wikiMap.last_refresh_job ?? wikiJob("succeeded", "completed"),
  );
}

describe("T58 Graph/Wiki durable job SSE", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: createLocalStorageMock(),
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("reconnects Graph with after_seq, ignores duplicate cursor events, and reconciles once on succeeded", async () => {
    const running = statusWithBuild("running", "projecting");
    const succeeded = statusWithBuild("succeeded", "completed");
    installBaseApiMocks({ status: running, wikiMap: idleWikiMap() });
    vi.mocked(api.getKnowledgeGraphStatus)
      .mockResolvedValueOnce(running)
      .mockResolvedValue(succeeded);

    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        sseResponse([
          sseFrame(1, "job_progress", {
            job_id: "kgjob-t58",
            job_kind: "graph",
            status: "running",
            stage: "projecting",
          }),
          sseFrame(2, "job_progress", {
            job_id: "kgjob-t58",
            job_kind: "graph",
            status: "running",
            stage: "persisting",
          }),
          sseFrame(2, "job_progress", {
            job_id: "kgjob-t58",
            job_kind: "graph",
            status: "running",
            stage: "duplicate",
          }),
        ]),
      )
      .mockResolvedValueOnce(
        sseResponse([
          sseFrame(2, "job_progress", {
            job_id: "kgjob-t58",
            job_kind: "graph",
            status: "running",
            stage: "duplicate_after_reconnect",
          }),
          sseFrame(3, "job_terminal", {
            job_id: "kgjob-t58",
            job_kind: "graph",
            status: "succeeded",
            stage: "completed",
          }),
          sseFrame(3, "job_terminal", {
            job_id: "kgjob-t58",
            job_kind: "graph",
            status: "succeeded",
            stage: "duplicate_terminal",
          }),
        ]),
      );
    vi.stubGlobal("fetch", fetchMock);
    const intervalSpy = vi.spyOn(window, "setInterval");

    renderGraph();

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2), { timeout: 3000 });
    await waitFor(() => expect(api.getKnowledgeGraphStatus).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(api.getKnowledgeGraph).toHaveBeenCalledTimes(2));

    const streamUrls = fetchMock.mock.calls.map(([input]) => requestUrl(input));
    expect(streamUrls[0]).toContain("/api/v1/knowledge-graph/jobs/kgjob-t58/stream");
    expect(streamUrls[1]).toContain("after_seq=2");
    expect(api.getKnowledgeGraphStatus).toHaveBeenCalledTimes(2);
    expect(api.getKnowledgeGraph).toHaveBeenCalledTimes(2);
    expect(api.getKnowledgeGraphSessionMap).toHaveBeenCalledTimes(2);
    expect(api.getKnowledgeGraphWikiMap).toHaveBeenCalledTimes(2);
    expect(usedLegacyPolling(intervalSpy)).toBe(false);
  });

  it("treats Wiki partial as terminal and performs exactly one job/map reconcile", async () => {
    const runningWiki = wikiMapWithJob("running", "generating");
    const partialWiki = wikiMapWithJob("partial", "partial");
    installBaseApiMocks({
      status: statusWithBuild("succeeded", "completed"),
      wikiMap: runningWiki,
    });
    vi.mocked(api.getKnowledgeGraphWikiMapJob).mockResolvedValue(
      partialWiki.last_refresh_job!,
    );
    vi.mocked(api.getKnowledgeGraphWikiMap)
      .mockResolvedValueOnce(runningWiki)
      .mockResolvedValue(partialWiki);

    const fetchMock = vi.fn<typeof fetch>().mockResolvedValueOnce(
      sseResponse([
        sseFrame(1, "job_progress", {
          job_id: "kgwikijob-t58",
          job_kind: "wiki",
          status: "running",
          stage: "generating",
        }),
        sseFrame(2, "job_terminal", {
          job_id: "kgwikijob-t58",
          job_kind: "wiki",
          status: "partial",
          stage: "partial",
        }),
        sseFrame(2, "job_terminal", {
          job_id: "kgwikijob-t58",
          job_kind: "wiki",
          status: "partial",
          stage: "duplicate_terminal",
        }),
      ]),
    );
    vi.stubGlobal("fetch", fetchMock);
    const intervalSpy = vi.spyOn(window, "setInterval");

    renderGraph();

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1), { timeout: 3000 });
    await waitFor(() => expect(api.getKnowledgeGraphWikiMapJob).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.getKnowledgeGraphWikiMap).toHaveBeenCalledTimes(2));

    expect(requestUrl(fetchMock.mock.calls[0][0])).toContain(
      "/api/v1/knowledge-graph/wiki-map/jobs/kgwikijob-t58/stream",
    );
    expect(api.getKnowledgeGraphWikiMapJob).toHaveBeenCalledTimes(1);
    expect(api.getKnowledgeGraphWikiMap).toHaveBeenCalledTimes(2);
    expect(usedLegacyPolling(intervalSpy)).toBe(false);
  });

  it("treats failed as terminal without reconnecting or repeated reconciliation", async () => {
    const running = statusWithBuild("running", "projecting");
    const failed = statusWithBuild("failed", "failed");
    installBaseApiMocks({ status: running, wikiMap: idleWikiMap() });
    vi.mocked(api.getKnowledgeGraphStatus)
      .mockResolvedValueOnce(running)
      .mockResolvedValue(failed);

    const fetchMock = vi.fn<typeof fetch>().mockResolvedValueOnce(
      sseResponse([
        sseFrame(1, "job_terminal", {
          job_id: "kgjob-t58",
          job_kind: "graph",
          status: "failed",
          stage: "failed",
          error_code: "INTERNAL_ERROR",
        }),
      ]),
    );
    vi.stubGlobal("fetch", fetchMock);
    const intervalSpy = vi.spyOn(window, "setInterval");

    renderGraph();

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1), { timeout: 3000 });
    await waitFor(() => expect(api.getKnowledgeGraphStatus).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(api.getKnowledgeGraph).toHaveBeenCalledTimes(2));

    await new Promise((resolve) => window.setTimeout(resolve, 50));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(api.getKnowledgeGraphStatus).toHaveBeenCalledTimes(2);
    expect(api.getKnowledgeGraph).toHaveBeenCalledTimes(2);
    expect(usedLegacyPolling(intervalSpy)).toBe(false);
  });
});
