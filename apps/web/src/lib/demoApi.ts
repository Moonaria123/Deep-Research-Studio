import type { Mode, Run, RunEvent } from "./types";
import publicSamples from "@/demo/public-samples.json";

export class ApiError extends Error {
  constructor(public code: string, message: string) { super(message); }
}

const now = 1_735_689_600;
const version = { id: "kg-demo-v1", workspace_id: "default", status: "succeeded", is_active: true, finished_at: now, activated_at: now, stats_json: { nodes: 8, edges: 9 } };
interface PublicSample { mode: Mode; query: string; status: string; result: Record<string, any> }
const sampleByMode = new Map((publicSamples as PublicSample[]).map((sample) => [sample.mode, sample]));
const fastSample = sampleByMode.get("fast")!;
const answer = fastSample.result.answer as Record<string, any>;
const sources: Record<string, any>[] = (fastSample.result.sources as Record<string, any>[]).filter((source) => source.url).slice(0, 3).map((source) => ({
  ...source,
  chunk_id: source.id,
  text: source.snippet,
}));
const makeRun = (mode: Mode, id = `demo-${mode}`, verification: "running" | "completed" = "completed"): Run => {
  const sample = sampleByMode.get(mode)!;
  const sampleResult = structuredClone(sample.result);
  sampleResult.sources = (sampleResult.sources ?? []).map((source: Record<string, any>) => ({
    ...source,
    chunk_id: source.id,
    text: source.snippet ?? "",
  }));
  sampleResult.run_id = id;
  sampleResult.mode = mode;
  sampleResult.status = "completed";
  sampleResult.verification = { ...sampleResult.verification, status: verification };
  sampleResult.confidence_warning = {
    ...(sampleResult.confidence_warning ?? { reasons: ["coverage_below_threshold"] }),
    acknowledged: false,
  };
  delete sampleResult.confidence_warning.acknowledged_at;
  if (verification === "running") delete sampleResult.verification.completed_at;
  return ({
  id, session_id: `session-${mode}`, mode, query: sample.query, status: "completed",
  created_at: now - 600, updated_at: now,
  result: sampleResult,
} as unknown as Run);
};

const runs = new Map<string, Run>(["fast", "multiple", "deep", "council"].map((mode) => [`demo-${mode}`, makeRun(mode as Mode)]));
runs.set("demo-regenerated", makeRun("fast", "demo-regenerated"));
const verificationUntil = new Map<string, number>();

const sessions = [
  { id: "session-fast", title: "SQLite FTS5 query operators", updated_at: now, pinned: true, latest_run_id: "demo-fast", latest_run_mode: "fast", latest_run_status: "completed" },
  { id: "session-multiple", title: "FTS5 query modes compared", updated_at: now - 3600, latest_run_id: "demo-multiple", latest_run_mode: "multiple", latest_run_status: "completed" },
  { id: "session-deep", title: "FTS5 indexing, tokenization and ranking", updated_at: now - 7200, latest_run_id: "demo-deep", latest_run_mode: "deep", latest_run_status: "completed" },
  { id: "session-council", title: "Model council: FTS5 for local research", updated_at: now - 10800, latest_run_id: "demo-council", latest_run_mode: "council", latest_run_status: "completed" },
];
const memories = [
  { id: "memory-1", layer: "workspace", namespace: "workspace", scope_key: "default", content: "Prefer the official SQLite FTS5 documentation for query syntax, tokenizers and ranking behavior.", salience: 1.5, effective_salience: 1.5, pinned: true, metadata: { source: "demo" } },
  { id: "memory-2", layer: "session", namespace: "session", scope_key: "session-deep", content: "Keep lexical FTS5 capabilities separate from optional semantic-vector retrieval.", salience: 1.1, effective_salience: 1.1, metadata: { source: "demo" } },
];
const documents = sources.map((source, index) => ({ id: source.document_id, url: source.url, title: source.title, content_type: "text/html", created_at: now - 86400 + index * 100, chunks: 6 + index * 2, origin: "external", artifact_type: "web_document", deletable: false }));
const reports = [{ id: "report-1", title: "SQLite FTS5 query operators", body_preview: answer.summary, body_md: answer.text, mode: "fast", session_id: "session-fast", run_id: "demo-fast", created_at: now }];
const notes = [{ id: "note-1", content: "Verify operator syntax against the official FTS5 page before relying on secondary tutorials.", created_at: now - 4000, session_id: "session-fast", run_id: "demo-fast" }];
const providerStatus = {
  llm_base_url: "", llm_model: "Preview only", has_llm_key: false, brave_ready: false, firecrawl_ready: false, embed_model: "Preview only", ollama_base_url: "", provider_config_revision: 1,
  search_providers: Object.fromEntries(["brave", "firecrawl", "tavily", "exa"].map((id) => [id, { configured: false, search_ready: false, extract_ready: false, enabled: false, runtime_eligible: false, selectable: false, credential_source: "none", probe_state: "not_configured", lamp: "gray" }])),
  external_search_service: { status: "unavailable", runtime_ready: false },
};
const registry = {
  providers: [{ id: "demo-preview", label: "Preview provider (disabled)", type: "openai", base_url: "https://example.invalid/v1", enabled: false, has_key: false, models: [{ id: "preview-model", label: "Preset output", capabilities: ["chat"], enabled: false }] }],
  roles: { default: "", embed: "", memory: "", council: [] },
};
const budget = { preset: "balanced", max_chars: 6000, turns: 6, recent_chars: 3500, semantic_chars: 1500, memory_chars: 1000, recent_share: 58, semantic_share: 25, memory_share: 17 };
const graphNodes = [
  { id: "node-session", type: "session", key: "session:session-deep", title: "FTS5 indexing and ranking", summary: "Research session connecting query syntax, tokenization and ranking.", attrs: { session_id: "session-deep", jump_uri: "/session/session-deep" }, confidence: .9, visible: true },
  { id: "node-source", type: "source", key: "source:sqlite", title: "SQLite FTS5 Extension", summary: "Primary evidence for FTS5 query and index behavior.", attrs: {}, confidence: .94, visible: true },
  { id: "node-report", type: "report", key: "report:fts5", title: "SQLite FTS5 query operators", summary: answer.summary, attrs: {}, confidence: .82, visible: true },
  { id: "node-memory", type: "memory", key: "memory:lexical", title: "Lexical versus semantic retrieval", summary: memories[0].content, attrs: {}, confidence: .78, visible: true },
];
const graphEdges = [
  { id: "edge-1", type: "uses", source: "node-session", target: "node-source", weight: 1, confidence: .9, visible: true, attrs: {} },
  { id: "edge-2", type: "produces", source: "node-session", target: "node-report", weight: 1, confidence: .9, visible: true, attrs: {} },
  { id: "edge-3", type: "remembers", source: "node-session", target: "node-memory", weight: .8, confidence: .8, visible: true, attrs: {} },
];

const readonly = () => Promise.reject(new ApiError("DEMO_READ_ONLY", "Preview only — changes are not saved."));
const handlers: Record<string, (...args: any[]) => any> = {
  health: async () => ({ status: "ok", fts5: true, db_encrypted: true, vector_backend: "numpy", job_backend: "rq", redis_reachable: true }),
  providers: async () => providerStatus, llmRegistry: async () => registry,
  listProviderModels: async () => ({ id: "demo-preview", models: registry.providers[0].models }),
  getSearchUsage: async () => ({ usage_mode: "auto", fallback_chain: ["brave", "tavily", "exa", "firecrawl"] }),
  estimateRunCost: async () => ({ mode: "fast", total_usd_low: 0, total_usd_high: 0, search_usd_low: 0, search_usd_high: 0, llm_usd_low: 0, llm_usd_high: 0, search_credits_low: 0, search_credits_high: 0, llm_input_tokens_est: 0, llm_output_tokens_est: 0, recovery_searches_max: 0, disclaimers: ["Preview fixture — no provider call"] }),
  enhancePrompt: async (query: string) => ({ enhanced: query }),
  listSessions: async () => ({ sessions }),
  listArchivedSessions: async () => ({ sessions: [{ id: "session-archive", title: "Archived FTS4 and FTS5 comparison", created_at: now - 200000, updated_at: now - 180000, archived_at: now - 170000, latest_run_mode: "multiple" }] }),
  listMessages: async (id: string) => ({ messages: [{ id: "msg-q", session_id: id, role: "user", content: fastSample.query, created_at: now - 700 }, { id: "msg-a", session_id: id, role: "assistant", content: answer.summary, created_at: now - 600, metadata: JSON.stringify({ run_id: id.replace("session", "demo"), mode: id.replace("session-", "") }) }] }),
  createRun: async (body: { mode: Mode }) => {
    const id = `demo-${body.mode}`;
    verificationUntil.set(id, Date.now() + 1400);
    return { run_id: id, session_id: `session-${body.mode}`, status: "completed" };
  },
  getRun: async (id: string) => {
    if (id === "demo-researching") return makeRun("deep", id, "running");
    const run = runs.get(id);
    if (!run) throw new ApiError("DEMO_NOT_FOUND", "This preview run does not exist.");
    const until = verificationUntil.get(id);
    if (until && Date.now() < until) return makeRun(run.mode, id, "running");
    verificationUntil.delete(id);
    return run;
  },
  acknowledgeRunConfidence: async () => ({ acknowledged: true, acknowledged_at: now }),
  regenerateRun: async () => {
    verificationUntil.set("demo-regenerated", Date.now() + 1400);
    return { run_id: "demo-regenerated", session_id: "session-fast", status: "completed", parent_run_id: "demo-fast" };
  },
  recallMemory: async () => ({ memories }), getMemory: async (id: string) => ({ memory: memories.find((m) => m.id === id) ?? memories[0] }), listArchivedMemories: async () => ({ memories: [] }),
  listSessionMemories: async () => ({ sessions: [{ session_id: "session-deep", session_title: "FTS5 indexing, tokenization and ranking", session_updated_at: now, memories: [memories[1]] }], total: 1, limit: 50, offset: 0, truncated: false }),
  getMemorySettings: async () => ({ post_turn_memory_refresh_enabled: true, workspace_memory_refresh_strategy: "incremental", memory_thinking_enabled: false, format_probe: null }),
  getMemoryWriteAudit: async () => ({ status: "completed", source: "demo_fixture", workspace_key: "default", user_key: "default", attempted: 2, succeeded: 2, skipped: 0, failed: 0, checked_at: now }),
  getMemoryDreamStatus: async () => ({ status: "idle", workspace_key: "default", user_key: "default", memories_created: 2, schedule: "weekly" }),
  previewMemoryOptimization: async () => ({ groups: [], threshold: .82 }),
  listLibraryDocuments: async () => ({ documents }), listDocuments: async () => ({ documents }),
  listLibraryReports: async () => ({ reports }), getLibraryReport: async () => ({ report: reports[0], provenance: [] }),
  listLibraryNotes: async () => ({ notes }), listNotes: async () => ({ notes }),
  listSourceSessions: async () => ({ artifacts: [{ session_id: "session-fast", run_id: "demo-fast", title: "SQLite FTS5 query operators" }] }),
  listSourceSnapshots: async () => ({ snapshots: [{ id: "snapshot-1", document_id: sources[0].document_id, url: sources[0].url, status: "ok", text: sources[0].text, captured_at: now }] }),
  getLibrarySnapshot: async () => ({ snapshot: { id: "snapshot-1", document_id: sources[0].document_id, url: sources[0].url, status: "ok", text: sources[0].text, captured_at: now } }),
  getKnowledgeGraphSettings: async () => ({ post_run_rebuild_enabled: true, wiki_auto_refresh_after_kg_rebuild: true }),
  getKnowledgeGraphStatus: async () => ({ workspace_id: "default", active_version: version, last_build_job: null, counts: { nodes: graphNodes.length, edges: graphEdges.length } }),
  getKnowledgeGraph: async () => ({ workspace_id: "default", active_version: version, nodes: graphNodes, edges: graphEdges }),
  getKnowledgeGraphNode: async (id: string) => ({ node: { ...(graphNodes.find((n) => n.id === id) ?? graphNodes[0]), edges: graphEdges, provenance: [] } }),
  getKnowledgeGraphSessionMap: async () => ({ workspace_id: "default", active_version: version, layout_mode: "session_tree", session_limit: 20, member_preview: 5, sessions: [{ session: { ...graphNodes[0], assistant_summary: answer.summary, assistant_excerpt: answer.summary, last_assistant_message_id: "msg-a" }, reports: [graphNodes[2]], notes: [], source_set: { count: 1, node_keys: [graphNodes[1].key], members: [graphNodes[1]] }, memory_set: { count: 1, node_keys: [graphNodes[3].key], members: [graphNodes[3]] }, counts: { reports: 1, notes: 0, sources: 1, memories: 1 } }], nodes_by_key: Object.fromEntries(graphNodes.map((n) => [n.key, n])), edges: graphEdges }),
  getKnowledgeGraphWikiMap: async () => ({ workspace_id: "default", layout_mode: "wiki_map", active_graph_version: version, active_snapshot: { id: "wiki-1", workspace_id: "default", graph_version_id: version.id, status: "succeeded", is_active: true, stale: false, stale_reason: "", finished_at: now, activated_at: now, stats: { topic_count: 3 } }, stale: false, stale_reason: "", stats: { topics: 3, tree_edges: 2, cross_links: 0 }, topics: [{ id: "topic-root", snapshot_id: "wiki-1", key: "wiki:root", title: "Research Wiki", summary: "Public SQLite FTS5 preview knowledge map.", depth: 0, parent_id: "", session_ids: [], source_message_ids: [], attrs: { kind: "workspace_root" } }, { id: "topic-session", snapshot_id: "wiki-1", key: "wiki:session:session-deep", title: "FTS5 indexing and ranking", summary: "Session topic root.", depth: 1, parent_id: "topic-root", session_ids: ["session-deep"], source_message_ids: ["msg-a"], attrs: { kind: "session_root", session_id: "session-deep", session_title: "FTS5 indexing and ranking" } }, { id: "topic-query", snapshot_id: "wiki-1", key: "wiki:topic:session-deep:query", title: "FTS5 query language", summary: "MATCH, phrase, prefix and NEAR queries.", depth: 2, parent_id: "topic-session", session_ids: ["session-deep"], source_message_ids: ["msg-a"], attrs: { kind: "topic", session_id: "session-deep", session_title: "FTS5 indexing and ranking", confidence: "high", quality: "public-sample" } }], tree_edges: [{ id: "wiki-edge-1", snapshot_id: "wiki-1", source: "topic-root", target: "topic-session", type: "parent_child", attrs: {} }, { id: "wiki-edge-2", snapshot_id: "wiki-1", source: "topic-session", target: "topic-query", type: "parent_child", attrs: {} }], cross_links: [], last_refresh_job: null }),
  getContextBudget: async () => ({ effective: budget, saved: null, env: null, default: budget, source: "default" }),
  recommendContextBudget: async () => ({ recommendation: budget, reasons: ["Balanced preview"] }),
  getDeepCheckpointSettings: async () => ({ auto_checkpoint_enabled: false }),
  getRetrievalSettings: async () => ({ saved: { chroma_persist_enabled: false }, effective: { chroma_persist_enabled: false, chroma_persist_mode: "ephemeral", chroma_persist_path: null, ephemeral_forced: false }, restart_required: false }),
  getPostRunNotes: async () => ({ enabled_modes: ["deep"] }),
  searchHistory: async () => ({ results: [], scope: "intra" }),
};

const mutationNames = new Set(["updateProviders", "saveSearchProviders", "toggleSearchProvider", "upsertLlmProvider", "removeLlmProvider", "patchLlmProvider", "createProviderModels", "updateProviderModel", "deleteProviderModel", "setLlmRoles", "testLlm", "testTools", "patchSession", "archiveSession", "restoreSession", "deleteSession", "updateContextBudget", "updateDeepCheckpointSettings", "updateSearchUsage", "updateRetrievalSettings", "updatePostRunNotes", "updateMemorySettings", "probeMemoryFormat", "updateKnowledgeGraphSettings", "refreshKnowledgeGraphWikiMap", "rebuildKnowledgeGraph", "purgeCache", "addMemory", "batchMemoryExpiry", "cleanupExpiredMemories", "archiveMemory", "restoreMemory", "patchMemory", "deleteMemory", "startMemoryDream", "cancelMemoryDream", "updateMemoryDreamSchedule", "applyMemoryOptimization", "indexUrl", "addNote", "addLibraryNote", "uploadLibraryReport", "importLibraryDocument", "deleteLibraryDocument", "cancelRun", "pauseRun", "continueRun", "abandonRunCheckpoint"]);
export const api = new Proxy({} as Record<string, (...args: any[]) => any>, {
  get: (_target, key: string) => handlers[key] ?? (mutationNames.has(key) ? readonly : () => Promise.reject(new ApiError("DEMO_UNSUPPORTED", `Preview fixture missing for ${key}.`))),
});

export function streamRun(runId: string, onEvent: (event: RunEvent) => void, onDone?: () => void): () => void {
  let cancelled = false;
  const timers = [
    window.setTimeout(() => !cancelled && onEvent({ seq: 1, type: "status", payload: { message: "Loading preset evidence…" } }), 120),
    window.setTimeout(() => !cancelled && onEvent({ seq: 2, type: "status", payload: { message: "Report ready; verification continues…" } }), 420),
    window.setTimeout(() => { if (!cancelled) { onEvent({ seq: 3, type: "done", payload: { run_id: runId } }); onDone?.(); } }, 850),
  ];
  return () => { cancelled = true; timers.forEach(window.clearTimeout); };
}

export function streamKnowledgeGraphJob(_kind: "graph" | "wiki", _id: string, _workspace: string, _onEvent: (event: any) => void, onDone?: () => void): () => void {
  const timer = window.setTimeout(() => onDone?.(), 200);
  return () => window.clearTimeout(timer);
}
