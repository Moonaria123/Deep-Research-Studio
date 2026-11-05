import type {
  ContextBudgetConfig,
  ContextBudgetRecommendation,
  ContextBudgetState,
  CouncilStrategy,
  DeepCheckpointContinueRequest,
  DeepCheckpointSettings,
  DeepCheckpointSummary,
  KnowledgeGraphBuildJob,
  KnowledgeGraphGraph,
  KnowledgeGraphJobEvent,
  KnowledgeGraphNodeDetail,
  KnowledgeGraphSessionMap,
  KnowledgeGraphSettings,
  KnowledgeGraphStatus,
  KnowledgeGraphWikiMap,
  KnowledgeGraphWikiRefreshJob,
  LibraryIndexWarning,
  LibraryDocument,
  LibraryDocumentDeleteResponse,
  LibraryProvenance,
  LibraryReport,
  LibraryScope,
  LocalDocumentImportResponse,
  LLMProviderUpsert,
  LLMRegistryStatus,
  LLMTestResult,
  MemoryDreamSchedule,
  MemoryDreamStatus,
  MemoryBatchExpiryResult,
  MemoryFormatProbeResult,
  MemoryItem,
  MemoryOptimizationPreview,
  MemorySettings,
  MemoryWriteAudit,
  SessionMemoryList,
  Message,
  Mode,
  ModelListResult,
  ModelProfile,
  ModelProfileInput,
  Note,
  PageSnapshot,
  PostRunNotesConfig,
  ProviderCandidate,
  ProviderStatus,
  ProviderUpdate,
  RetrievalSettings,
  Run,
  RunCostEstimate,
  RunCostEstimateRequest,
  RunEvent,
  SearchOptions,
  SearchProviderId,
  SearchProviderMutationResponse,
  SearchProviderSaveRequest,
  SearchUsagePolicy,
  Session,
  ArchivedSession,
  ThinkingEffort,
  ToolsTestResult,
} from "./types";
import { API_BASE } from "./apiOrigin";

const BASE = API_BASE;

export class ApiError extends Error {
  code: string;
  constructor(code: string, message: string) {
    super(message);
    this.code = code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    const err = data?.error ?? {};
    throw new ApiError(err.code ?? "UNKNOWN", err.message ?? response.statusText);
  }
  return data as T;
}

async function requestForm<T>(path: string, body: FormData): Promise<T> {
  const response = await fetch(`${BASE}${path}`, { method: "POST", body });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    const err = data?.error ?? {};
    throw new ApiError(err.code ?? "UNKNOWN", err.message ?? response.statusText);
  }
  return data as T;
}

export const api = {
  health: () =>
    request<{
      status: string;
      fts5: boolean;
      db_encrypted?: boolean;
      auth_mode?: "off" | "token";
      vector_backend?: "bruteforce" | "numpy" | "chroma" | null;
      chroma_persist_enabled?: boolean;
      chroma_persist_mode?: "disk" | "ephemeral";
      chroma_persist_path?: string | null;
      job_backend?: "rq" | "thread";
      redis_reachable?: boolean | null;
      context_budget_preset?: string;
      post_run_notes_modes?: Mode[];
    }>("/health"),
  providers: () => request<ProviderStatus>("/providers"),
  updateProviders: (body: ProviderUpdate) =>
    request<ProviderStatus>("/providers", { method: "PUT", body: JSON.stringify(body) }),
  saveSearchProviders: (body: SearchProviderSaveRequest) =>
    request<SearchProviderMutationResponse>("/settings/search-providers/save", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  toggleSearchProvider: (body: { expected_revision: number; provider: SearchProviderId; enabled: boolean }) =>
    request<SearchProviderMutationResponse>("/settings/search-providers/enabled", {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  llmRegistry: () => request<LLMRegistryStatus>("/providers/llm"),
  upsertLlmProvider: (body: LLMProviderUpsert) =>
    request<LLMRegistryStatus>("/providers/llm", { method: "PUT", body: JSON.stringify(body) }),
  removeLlmProvider: (id: string) =>
    request<LLMRegistryStatus>(`/providers/llm/${encodeURIComponent(id)}`, { method: "DELETE" }),
  patchLlmProvider: (id: string, body: Partial<LLMProviderUpsert>) =>
    request<LLMRegistryStatus>(`/providers/llm/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  listProviderModels: (id: string) =>
    request<{ id: string; models: ModelProfile[] }>(
      `/providers/llm/${encodeURIComponent(id)}/models`,
    ),
  createProviderModels: (id: string, body: { models: ModelProfileInput[] }) =>
    request<{ id: string; added: number; models: ModelProfile[] }>(
      `/providers/llm/${encodeURIComponent(id)}/models`,
      { method: "POST", body: JSON.stringify(body) },
    ),
  updateProviderModel: (id: string, modelId: string, body: Partial<ModelProfileInput>) =>
    request<{ id: string; models: ModelProfile[] }>(
      `/providers/llm/${encodeURIComponent(id)}/models/${encodeURIComponent(modelId)}`,
      { method: "PATCH", body: JSON.stringify(body) },
    ),
  deleteProviderModel: (id: string, modelId: string) =>
    request<LLMRegistryStatus>(
      `/providers/llm/${encodeURIComponent(id)}/models/${encodeURIComponent(modelId)}`,
      { method: "DELETE" },
    ),
  setLlmRoles: (body: { default?: string; embed?: string; memory?: string; council?: string[] }) =>
    request<LLMRegistryStatus>("/providers/roles", { method: "PUT", body: JSON.stringify(body) }),
  fetchModels: (body: ProviderCandidate) =>
    request<ModelListResult>("/providers/models", { method: "POST", body: JSON.stringify(body) }),
  testLlm: (body: ProviderCandidate) =>
    request<LLMTestResult>("/providers/test", { method: "POST", body: JSON.stringify(body) }),
  testTools: (body: {
    provider?: SearchProviderId;
    brave_api_key?: string;
    firecrawl_api_key?: string;
    firecrawl_base_url?: string;
    exa_api_key?: string;
    tavily_api_key?: string;
  }) =>
    request<ToolsTestResult>("/tools/test", { method: "POST", body: JSON.stringify(body) }),
  listSessions: () => request<{ sessions: Session[] }>("/sessions"),
  listArchivedSessions: () => request<{ sessions: ArchivedSession[] }>("/sessions/archived"),
  patchSession: (
    sessionId: string,
    body: { pinned?: boolean; mode_default?: Mode | null; search_usage_policy?: SearchUsagePolicy | null },
  ) =>
    request<{ session_id: string; session: Session }>(`/sessions/${encodeURIComponent(sessionId)}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  archiveSession: (sessionId: string) =>
    request<{ session_id: string; archived: boolean }>(`/sessions/${sessionId}/archive`, {
      method: "POST",
    }),
  restoreSession: (sessionId: string) =>
    request<{ session_id: string; archived: boolean }>(`/sessions/${sessionId}/restore`, {
      method: "POST",
    }),
  deleteSession: (sessionId: string) =>
    request<{ session_id: string; deleted: boolean }>(`/sessions/${sessionId}`, { method: "DELETE" }),
  listMessages: (sessionId: string) =>
    request<{ messages: Message[] }>(`/sessions/${sessionId}/messages`),
  createRun: (body: {
    query: string;
    mode: Mode;
    session_id?: string;
    budget?: Record<string, number>;
    llm_model?: string;
    council_models?: string[];
    council_size?: number;
    council_strategy?: CouncilStrategy;
    temperature?: number;
    thinking_effort?: ThinkingEffort;
    memory_policy?: "off" | "workspace" | "workspace_user";
    workspace_key?: string;
    user_key?: string;
    library_scope?: LibraryScope;
    context_budget?: Partial<ContextBudgetConfig>;
    search_options?: SearchOptions;
    search_usage_policy?: SearchUsagePolicy | null;
  }) =>
    request<{ run_id: string; session_id: string; status: string }>("/runs", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  estimateRunCost: (body: RunCostEstimateRequest) =>
    request<RunCostEstimate>("/runs/estimate-cost", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  getContextBudget: () => request<ContextBudgetState>("/settings/context-budget"),
  getDeepCheckpointSettings: () =>
    request<DeepCheckpointSettings>("/settings/deep-checkpoint"),
  updateDeepCheckpointSettings: (body: DeepCheckpointSettings) =>
    request<DeepCheckpointSettings>("/settings/deep-checkpoint", {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  getSearchUsage: () => request<SearchUsagePolicy>("/settings/search-usage"),
  updateSearchUsage: (body: SearchUsagePolicy) =>
    request<SearchUsagePolicy>("/settings/search-usage", {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  updateContextBudget: (body: Partial<ContextBudgetConfig>) =>
    request<ContextBudgetState>("/settings/context-budget", {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  getRetrievalSettings: () => request<RetrievalSettings>("/settings/retrieval"),
  updateRetrievalSettings: (body: Partial<RetrievalSettings["saved"]>) =>
    request<RetrievalSettings>("/settings/retrieval", {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  getPostRunNotes: () => request<PostRunNotesConfig>("/settings/post-run-notes"),
  updatePostRunNotes: (body: PostRunNotesConfig) =>
    request<PostRunNotesConfig>("/settings/post-run-notes", {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  getMemorySettings: () => request<MemorySettings>("/settings/memory"),
  updateMemorySettings: (body: Partial<MemorySettings>) =>
    request<MemorySettings>("/settings/memory", {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  probeMemoryFormat: () =>
    request<MemoryFormatProbeResult>("/settings/memory/format-probe", {
      method: "POST",
    }),
  getMemoryWriteAudit: (params: { workspace_key?: string; user_key?: string } = {}) => {
    const q = new URLSearchParams();
    if (params.workspace_key) q.set("workspace_key", params.workspace_key);
    if (params.user_key) q.set("user_key", params.user_key);
    const query = q.toString();
    return request<MemoryWriteAudit>(`/memory/write-audit${query ? `?${query}` : ""}`);
  },
  getKnowledgeGraphSettings: () => request<KnowledgeGraphSettings>("/settings/knowledge-graph"),
  updateKnowledgeGraphSettings: (body: KnowledgeGraphSettings) =>
    request<KnowledgeGraphSettings>("/settings/knowledge-graph", {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  getKnowledgeGraphStatus: (params: { workspace_id?: string } = {}) => {
    const q = new URLSearchParams();
    if (params.workspace_id) q.set("workspace_id", params.workspace_id);
    const query = q.toString();
    return request<KnowledgeGraphStatus>(`/knowledge-graph/status${query ? `?${query}` : ""}`);
  },
  getKnowledgeGraphJob: (
    jobId: string,
    params: { workspace_id?: string } = {},
  ) => {
    const q = new URLSearchParams();
    if (params.workspace_id) q.set("workspace_id", params.workspace_id);
    const query = q.toString();
    return request<KnowledgeGraphBuildJob>(
      `/knowledge-graph/jobs/${encodeURIComponent(jobId)}${query ? `?${query}` : ""}`,
    );
  },
  getKnowledgeGraph: (params: { workspace_id?: string; limit?: number; edge_types?: string[] } = {}) => {
    const q = new URLSearchParams();
    if (params.workspace_id) q.set("workspace_id", params.workspace_id);
    if (params.limit) q.set("limit", String(params.limit));
    if (params.edge_types?.length) q.set("edge_types", params.edge_types.join(","));
    const query = q.toString();
    return request<KnowledgeGraphGraph>(`/knowledge-graph/graph${query ? `?${query}` : ""}`);
  },
  getKnowledgeGraphSessionMap: (
    params: { workspace_id?: string; session_limit?: number; member_preview?: number } = {},
  ) => {
    const q = new URLSearchParams();
    if (params.workspace_id) q.set("workspace_id", params.workspace_id);
    if (params.session_limit) q.set("session_limit", String(params.session_limit));
    if (typeof params.member_preview === "number") q.set("member_preview", String(params.member_preview));
    const query = q.toString();
    return request<KnowledgeGraphSessionMap>(`/knowledge-graph/session-map${query ? `?${query}` : ""}`);
  },
  getKnowledgeGraphWikiMap: (params: { workspace_id?: string } = {}) => {
    const q = new URLSearchParams();
    if (params.workspace_id) q.set("workspace_id", params.workspace_id);
    const query = q.toString();
    return request<KnowledgeGraphWikiMap>(`/knowledge-graph/wiki-map${query ? `?${query}` : ""}`);
  },
  refreshKnowledgeGraphWikiMap: (body: { workspace_id?: string } = {}) =>
    request<KnowledgeGraphWikiRefreshJob>("/knowledge-graph/wiki-map/refresh", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  getKnowledgeGraphWikiMapJob: (
    jobId: string,
    params: { workspace_id?: string } = {},
  ) => {
    const q = new URLSearchParams();
    if (params.workspace_id) q.set("workspace_id", params.workspace_id);
    const query = q.toString();
    return request<KnowledgeGraphWikiRefreshJob>(
      `/knowledge-graph/wiki-map/jobs/${encodeURIComponent(jobId)}${query ? `?${query}` : ""}`,
    );
  },
  getKnowledgeGraphNode: (nodeId: string) =>
    request<{ node: KnowledgeGraphNodeDetail }>(`/knowledge-graph/nodes/${encodeURIComponent(nodeId)}`),
  rebuildKnowledgeGraph: (body: { workspace_id?: string } = {}) =>
    request<KnowledgeGraphStatus["last_build_job"]>("/knowledge-graph/rebuild", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  purgeCache: (body: { scope?: "all" | "search" | "fetch" } = {}) => {
    const q = new URLSearchParams();
    q.set("scope", body.scope ?? "all");
    return request<{ purged_count: number; scope: "all" | "search" | "fetch" }>(`/cache?${q.toString()}`, {
      method: "DELETE",
    });
  },
  recommendContextBudget: (body: {
    session_id?: string;
    memory_policy?: string;
    workspace_key?: string;
  }) =>
    request<ContextBudgetRecommendation>("/context-budget/recommend", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  getRun: (runId: string) => request<Run>(`/runs/${runId}`),
  acknowledgeRunConfidence: (runId: string) =>
    request<{ acknowledged: boolean; acknowledged_at?: number }>(
      `/runs/${encodeURIComponent(runId)}/confidence/acknowledge`,
      { method: "POST" },
    ),
  regenerateRun: (runId: string) =>
    request<{ run_id: string; session_id?: string; status: string; parent_run_id?: string }>(
      `/runs/${encodeURIComponent(runId)}/regenerate`,
      { method: "POST" },
    ),
  cancelRun: (runId: string) =>
    request<{ run_id: string; status: string }>(`/runs/${runId}/cancel`, { method: "POST" }),
  pauseRun: (runId: string) =>
    request<{ run_id: string; status: string; accepted: boolean }>(`/runs/${runId}/pause`, {
      method: "POST",
    }),
  getRunCheckpoint: (runId: string) =>
    request<DeepCheckpointSummary>(`/runs/${runId}/checkpoint`),
  continueRun: (runId: string, body: DeepCheckpointContinueRequest) =>
    request<{
      run_id: string;
      session_id: string;
      status: string;
      parent_run_id: string;
      idempotent_replay: boolean;
    }>(`/runs/${runId}/continue`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  abandonRunCheckpoint: (runId: string, expectedCheckpointRevision: number) =>
    request<DeepCheckpointSummary>(`/runs/${runId}/checkpoint/abandon`, {
      method: "POST",
      body: JSON.stringify({ expected_checkpoint_revision: expectedCheckpointRevision }),
    }),
  recallMemory: (params: {
    session_key?: string;
    workspace_key?: string;
    user_key?: string;
    limit?: number;
    query?: string;
  }) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v != null && q.set(k, String(v)));
    return request<{ memories: MemoryItem[] }>(`/memories?${q.toString()}`);
  },
  searchHistory: (
    sessionId: string,
    params: { query: string; scope?: "intra" | "cross"; limit?: number },
  ) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v != null && q.set(k, String(v)));
    return request<{ results: Message[]; scope: string }>(
      `/sessions/${sessionId}/history/search?${q.toString()}`,
    );
  },
  addMemory: (body: { layer: string; scope_key: string; content: string; salience?: number }) =>
    request<{ memory_id: string }>("/memories", { method: "POST", body: JSON.stringify(body) }),
  getMemory: (memoryId: string) =>
    request<{ memory: MemoryItem }>(`/memories/${encodeURIComponent(memoryId)}`),
  listArchivedMemories: (params: { workspace_key?: string; user_key?: string; limit?: number } = {}) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v != null && q.set(k, String(v)));
    return request<{ memories: MemoryItem[] }>(`/memories/archived?${q.toString()}`);
  },
  listSessionMemories: (params: {
    workspace_key?: string;
    session_id?: string;
    query?: string;
    include_archived_sessions?: boolean;
    include_archived_memories?: boolean;
    limit?: number;
    offset?: number;
  } = {}) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v != null && q.set(k, String(v)));
    return request<SessionMemoryList>(`/memories/session?${q.toString()}`);
  },
  batchMemoryExpiry: (body: {
    workspace_key?: string;
    user_key?: string;
    namespace: "workspace" | "user";
    expires_at: number | null;
    dry_run: boolean;
  }) =>
    request<MemoryBatchExpiryResult>("/memories/batch-expiry", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  cleanupExpiredMemories: (
    body:
      | { namespace: "workspace" | "user"; scope_key: string; limit?: number }
      | { namespace?: never; scope_key?: never; workspace_key?: string; user_key?: string; limit?: number } = {},
  ) =>
    request<{ archived: number; memory_ids: string[]; remaining: number }>("/memories/cleanup", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  archiveMemory: (memoryId: string) =>
    request<{ memory_id: string; archived: boolean }>(`/memories/${encodeURIComponent(memoryId)}/archive`, {
      method: "POST",
    }),
  restoreMemory: (memoryId: string) =>
    request<{ memory_id: string; archived: boolean; library_index_warning: LibraryIndexWarning | null }>(
      `/memories/${encodeURIComponent(memoryId)}/restore`,
      {
        method: "POST",
      },
    ),
  patchMemory: (
    memoryId: string,
    body: {
      content?: string;
      salience?: number;
      metadata?: Record<string, unknown>;
      pinned?: boolean;
      expires_at?: number | null;
    },
  ) =>
    request<{ memory_id: string; memory: MemoryItem; library_index_warning: LibraryIndexWarning | null }>(
      `/memories/${encodeURIComponent(memoryId)}`,
      {
        method: "PATCH",
        body: JSON.stringify(body),
      },
    ),
  deleteMemory: (memoryId: string) =>
    request<{ memory_id: string; deleted: boolean }>(`/memories/${encodeURIComponent(memoryId)}`, {
      method: "DELETE",
    }),
  previewMemoryOptimization: (body: { workspace_key?: string; user_key?: string; threshold?: number; limit?: number } = {}) =>
    request<MemoryOptimizationPreview>("/memories/optimize/preview", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  applyMemoryOptimization: (body: { strategy: "merge" | "archive" | "downrank"; memory_ids: string[]; factor?: number }) =>
    request<{
      strategy: string;
      memory_id?: string;
      archived_ids?: string[];
      memory_ids?: string[];
      summary?: string;
      library_index_warning?: LibraryIndexWarning | null;
    }>(
      "/memories/optimize/apply",
      { method: "POST", body: JSON.stringify(body) },
    ),
  getMemoryDreamStatus: (params: { workspace_key?: string; user_key?: string } = {}) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v != null && q.set(k, String(v)));
    return request<MemoryDreamStatus>(`/memory/dream?${q.toString()}`);
  },
  startMemoryDream: (body: { workspace_key?: string; user_key?: string } = {}) =>
    request<MemoryDreamStatus>("/memory/dream", { method: "POST", body: JSON.stringify(body) }),
  cancelMemoryDream: (dreamRunId: string) =>
    request<MemoryDreamStatus>(`/memory/dream/${encodeURIComponent(dreamRunId)}/cancel`, { method: "POST" }),
  updateMemoryDreamSchedule: (body: { schedule: MemoryDreamSchedule }) =>
    request<{ schedule: MemoryDreamSchedule; next_scheduled_at?: number | null }>("/memory/dream/schedule", {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  indexUrl: (url: string) =>
    request<{ document_id: string; chunks: number; embedded: boolean }>("/index/url", {
      method: "POST",
      body: JSON.stringify({ url }),
    }),
  enhancePrompt: (query: string) =>
    request<{ enhanced: string }>("/tools/enhance", { method: "POST", body: JSON.stringify({ query }) }),
  listLibraryDocuments: () => request<{ documents: LibraryDocument[] }>("/library/documents"),
  importLibraryDocument: (file: File, title: string) => {
    const body = new FormData();
    body.append("title", title);
    body.append("file", file);
    return requestForm<LocalDocumentImportResponse>("/library/documents/import", body);
  },
  deleteLibraryDocument: (documentId: string) =>
    request<LibraryDocumentDeleteResponse>(
      `/library/documents/${encodeURIComponent(documentId)}`,
      { method: "DELETE" },
    ),
  listLibraryNotes: () => request<{ notes: Note[] }>("/library/notes"),
  addLibraryNote: (body: { content: string; session_id?: string; run_id?: string }) =>
    request<{ note_id: string }>("/library/notes", { method: "POST", body: JSON.stringify(body) }),
  listLibraryReports: () => request<{ reports: LibraryReport[] }>("/library/reports"),
  getLibraryReport: (reportId: string) => request<{ report: LibraryReport }>(`/library/reports/${reportId}`),
  uploadLibraryReport: (file: File, title: string) => {
    const body = new FormData();
    body.append("title", title);
    body.append("file", file);
    return requestForm<{ report_id: string }>("/library/reports/upload", body);
  },
  listSourceSessions: (documentId: string) =>
    request<{ artifacts: LibraryProvenance[] }>(`/library/sources/${documentId}/sessions`),
  listSourceSnapshots: (documentId: string) =>
    request<{ snapshots: PageSnapshot[] }>(`/library/sources/${documentId}/snapshots`),
  getLibrarySnapshot: (snapshotId: string) =>
    request<{ snapshot: PageSnapshot }>(`/library/snapshots/${snapshotId}`),
  listDocuments: () => request<{ documents: LibraryDocument[] }>("/index/documents"),
  listNotes: () => request<{ notes: Note[] }>("/notes"),
  addNote: (content: string) =>
    request<{ note_id: string }>("/notes", { method: "POST", body: JSON.stringify({ content }) }),
};

/**
 * Stream a run's Server-Sent Events. Parses the text/event-stream body incrementally
 * and invokes onEvent for each event. Returns a function to abort the stream.
 */
export function streamRun(
  runId: string,
  onEvent: (event: RunEvent) => void,
  onDone?: () => void,
  onError?: (error: Error) => void,
  afterSeq = 0,
): () => void {
  const controller = new AbortController();
  (async () => {
    try {
      const cursor = Math.max(0, Math.trunc(afterSeq));
      const suffix = cursor > 0 ? `?after_seq=${cursor}` : "";
      const response = await fetch(`${BASE}/runs/${runId}/stream${suffix}`, { signal: controller.signal });
      if (!response.ok) {
        const text = await response.text();
        const data = text ? JSON.parse(text) : {};
        const err = data?.error ?? {};
        throw new ApiError(err.code ?? "UNKNOWN", err.message ?? response.statusText);
      }
      if (!response.body) throw new Error("no stream body");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let seq = cursor;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";
        for (const frame of frames) {
          const typeMatch = frame.match(/^event: (.*)$/m);
          const dataMatch = frame.match(/^data: (.*)$/m);
          if (!dataMatch) continue;
          const type = typeMatch?.[1] ?? "message";
          const payload = JSON.parse(dataMatch[1]);
          const idMatch = frame.match(/^id: (\d+)$/m);
          const persistedSeq = idMatch ? Number(idMatch[1]) : seq + 1;
          seq = Math.max(seq, persistedSeq);
          onEvent({ seq: persistedSeq, type, payload });
        }
      }
      onDone?.();
    } catch (error) {
      if ((error as Error).name !== "AbortError") onError?.(error as Error);
    }
  })();
  return () => controller.abort();
}

/** Stream one durable Graph/Wiki maintenance job from an authoritative cursor. */
export function streamKnowledgeGraphJob(
  jobKind: "graph" | "wiki",
  jobId: string,
  workspaceId: string,
  onEvent: (event: KnowledgeGraphJobEvent) => void,
  onDone?: () => void,
  onError?: (error: Error) => void,
  afterSeq = 0,
): () => void {
  const controller = new AbortController();
  (async () => {
    try {
      const cursor = Math.max(0, Math.trunc(afterSeq));
      const query = new URLSearchParams({ workspace_id: workspaceId || "default" });
      if (cursor > 0) query.set("after_seq", String(cursor));
      const route =
        jobKind === "graph"
          ? `/knowledge-graph/jobs/${encodeURIComponent(jobId)}/stream`
          : `/knowledge-graph/wiki-map/jobs/${encodeURIComponent(jobId)}/stream`;
      const response = await fetch(`${BASE}${route}?${query.toString()}`, {
        signal: controller.signal,
      });
      if (!response.ok) {
        const text = await response.text();
        const data = text ? JSON.parse(text) : {};
        const err = data?.error ?? {};
        throw new ApiError(err.code ?? "UNKNOWN", err.message ?? response.statusText);
      }
      if (!response.body) throw new Error("no stream body");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
        const frames = buffer.replace(/\r\n/g, "\n").split("\n\n");
        buffer = frames.pop() ?? "";
        for (const frame of frames) {
          const idMatch = frame.match(/^id:\s*(\d+)$/m);
          const typeMatch = frame.match(/^event:\s*(.*)$/m);
          const dataLines = frame
            .split("\n")
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trimStart());
          if (!idMatch || !dataLines.length) continue;
          const payload = JSON.parse(dataLines.join("\n"));
          if (
            !payload
            || typeof payload !== "object"
            || payload.job_id !== jobId
            || payload.job_kind !== jobKind
          ) {
            continue;
          }
          onEvent({
            seq: Number(idMatch[1]),
            type: typeMatch?.[1]?.trim() || "message",
            payload,
          });
        }
        if (done) break;
      }
      onDone?.();
    } catch (error) {
      if ((error as Error).name !== "AbortError") onError?.(error as Error);
    }
  })();
  return () => controller.abort();
}
