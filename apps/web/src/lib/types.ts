export type Mode = "fast" | "multiple" | "deep" | "council";

export type CouncilStrategy = "parallel_vote" | "judge_synthesis" | "debate_once";

export type LibraryScope = "off" | "notes" | "reports" | "all";

export type EvidenceOrigin = "external" | "local" | "memory";
export type EvidenceArtifactType =
  | "web_document"
  | "library_note"
  | "library_report"
  | "library_document"
  | "memory";
export type EvidenceSourceStatus = "active" | "archived";

export type SearchFreshness = "pd" | "pw" | "pm" | "py";

export interface SearchOptions {
  include_domains?: string[];
  exclude_domains?: string[];
  freshness?: SearchFreshness;
  extra_subqueries?: string[];
  country?: string;
  search_lang?: string;
  ui_language?: string;
}

export interface SearchLocaleSnapshot {
  country?: string;
  search_lang?: string;
  ui_language?: string;
}

export type SearchProviderId = "brave" | "tavily" | "exa" | "firecrawl";
export type SearchUsageMode = "single" | "dual_role" | "auto";

export interface SearchUsagePolicy {
  usage_mode: SearchUsageMode;
  primary_provider?: SearchProviderId;
  search_provider?: SearchProviderId;
  extract_provider?: Exclude<SearchProviderId, "brave">;
  fallback_chain: SearchProviderId[];
}

export interface ResolvedSearchUsage {
  source: "global" | "session";
  effective_policy: SearchUsagePolicy;
  search_chain: SearchProviderId[];
  extract_chain: Exclude<SearchProviderId, "brave">[];
  search_fallback_tail?: ["cache", "indexed_external"];
  reason_codes: string[];
  provider_config_revision?: number;
  credential_revision_snapshot?: Partial<Record<SearchProviderId, number>>;
  external_search_status?: ExternalSearchServiceStatus;
}

export interface SearchProviderCapability {
  configured: boolean;
  search_ready: boolean;
  extract_ready: boolean;
  provider?: SearchProviderId;
  credential_source?: "none" | "config" | "secure" | "env";
  credential_revision?: number | null;
  enabled?: boolean;
  probe_state?:
    | "not_configured"
    | "not_tested"
    | "stale"
    | "available"
    | "rate_limited"
    | "action_required"
    | "service_unavailable"
    | "network_unreachable"
    | "credential_rejected";
  credential_state?: "accepted" | "rejected" | "unknown";
  availability?: "available" | "temporarily_unavailable" | "action_required" | "unavailable" | "unknown";
  lamp?: "green" | "yellow" | "red" | "gray";
  checked_at?: number | null;
  stale_at?: number | null;
  retry_after?: number | null;
  probe_credential_revision?: number | null;
  runtime_eligible?: boolean;
  selectable?: boolean;
  reason_codes?: string[];
  draft_accepted?: boolean;
  stored_key_retained?: boolean;
  probe_persisted?: boolean;
}

export interface ExternalSearchServiceStatus {
  status: "ready" | "warn" | "unavailable";
  runtime_ready: boolean;
  provider_ready_count?: number;
  reason_codes?: string[];
  banner?: { code: string; severity: "info" | "warning" | "error"; message: string };
}

export interface SearchProviderSaveRequest {
  expected_revision: number;
  providers: Partial<Record<SearchProviderId, {
    api_key?: string;
    base_url?: string;
    remove?: boolean;
  }>>;
}

export interface SearchProviderMutationResponse {
  operation_status: "success" | "partial" | "rejected" | "noop" | "conflict";
  revision: number;
  providers: Record<SearchProviderId, SearchProviderCapability>;
  saved?: SearchProviderId[];
  rejected?: SearchProviderId[];
  external_search_service: ExternalSearchServiceStatus;
  banner?: { code: string; severity: "info" | "warning" | "error"; message: string };
}

export type AnswerResetReason =
  | "provider_stream_failed"
  | "council_judge_fallback"
  | "draft_invalidated"
  | "cancelled"
  | "deadline_exceeded";

export interface AnswerResetPayload {
  generation_id: string;
  reason: AnswerResetReason;
}

export interface PhaseMetric {
  plan_id?: string;
  lane?: EvidenceOrigin;
  query?: string;
  stage: string;
  provider?: string;
  attempt: number;
  started_at: number;
  finished_at: number;
  duration_ms: number;
  status: string;
  deadline_remaining_before_ms?: number | null;
  deadline_remaining_after_ms?: number | null;
  evidence_id?: string | null;
  candidate_id?: string | null;
}

export interface DeadlineSummary {
  requested_at: number;
  deadline_at: number;
  worker_started_at: number | null;
  writer_started_at: number | null;
  terminal_at: number | null;
  queue_delay_ms: number;
  writer_reserve_s: number;
  finalize_reserve_s: number;
  cleanup_reserve_s: number;
  deadline_met: boolean;
  cleanup_overrun_ms: number;
}

export interface RetrievalBudget {
  results_per_query: number;
  candidate_count: number;
  fetches_max: number;
  fetches_used: number;
  fetch_concurrency: number;
  writer_reserve_s: number;
  finalize_reserve_s: number;
  cleanup_reserve_s: number;
  recovery_fetches_max: number;
  recovery_fetches_used: number;
  rejection_reason_codes?: string[];
}

export interface RetrievalLocaleOutcome {
  requested: SearchLocaleSnapshot;
  effective: SearchLocaleSnapshot;
  applied_fields: (keyof SearchLocaleSnapshot)[];
  ignored_fields: (keyof SearchLocaleSnapshot)[];
  reason_codes: string[];
}

export interface RetrievalSearchOptionsOutcome {
  requested: Pick<SearchOptions, "include_domains" | "exclude_domains" | "freshness">;
  applied_fields: ("include_domains" | "exclude_domains" | "freshness")[];
  ignored_fields: ("include_domains" | "exclude_domains" | "freshness")[];
  reason_codes: string[];
}

export interface StreamMetrics {
  first_answer_delta_ms?: number | null;
  pre_writer_ms?: number | null;
  provider_first_delta_ms?: number | null;
  delta_event_count: number;
  delta_bytes: number;
  sync_fallback_used: boolean;
}

export interface WriterAttemptDiagnostic {
  attempt: number;
  provider?: string | null;
  model?: string | null;
  call_mode?: "stream" | "sync" | string;
  classification?: string;
  status?: string;
  fallback_scope?: "none" | "same_provider_sync" | "cross_backend" | "sync_only" | string;
  finish_reason?: string | null;
  visible_chars?: number;
  usage?: Record<string, string | number | boolean | null>;
  call_options?: LLMCallOptionsDiagnostic;
  generation_id?: string;
}

export interface LLMCallOptionsDiagnostic {
  provider_identity?: string;
  provider_family?: string;
  endpoint_type?: string;
  reasoning_transport?: "top_level" | "extra_body" | "none" | string;
  structured_output_strategy?: "native" | "prompt_validated" | string;
  output_token_parameter?: string;
  shared_output_cap?: boolean;
  thinking_target?: string | null;
  visible_output_target?: number | null;
  reasoning_reserve_tokens?: number | null;
  provider_output_cap?: number | null;
  requested?: Record<string, string | number | boolean | null>;
  applied?: Record<string, string | number | boolean | null>;
  ignored?: Record<string, string | number | boolean | null>;
  reason_codes?: string[];
}

export interface WriterDiagnostics {
  classification: string;
  provider?: string | null;
  model?: string | null;
  finish_reason?: string | null;
  visible_chars: number;
  usage?: Record<string, string | number | boolean | null>;
  fallback_scope?: string;
  sync_fallback_used?: boolean;
  attempts?: WriterAttemptDiagnostic[];
  call_options?: LLMCallOptionsDiagnostic;
  writer_started_at?: number | null;
  deadline_at?: number | null;
  generation_id?: string | null;
}

export interface RunCostEstimateRequest {
  mode: Mode;
  budget?: Record<string, number>;
  llm_model?: string;
  council_models?: string[];
  council_size?: number;
  council_strategy?: CouncilStrategy;
  search_options?: SearchOptions;
}

export interface RunCostEstimate {
  mode: Mode | string;
  searches: number;
  recovery_searches_max?: number;
  search_usd_low: number | null;
  search_usd_high: number | null;
  search_credits_low: number | null;
  search_credits_high: number | null;
  llm_input_tokens_est: number;
  llm_output_tokens_est: number;
  llm_usd_low: number | null;
  llm_usd_high: number | null;
  total_usd_low: number | null;
  total_usd_high: number | null;
  confidence: "medium" | "low" | "unknown" | string;
  pricing_basis: Record<string, unknown>;
  disclaimers: string[];
}

export interface RetrievalSettings {
  saved: {
    chroma_persist_enabled: boolean;
    kg_aware_retrieval_enabled?: boolean;
  };
  effective: {
    chroma_persist_enabled: boolean;
    chroma_persist_mode: "disk" | "ephemeral";
    chroma_persist_path: string | null;
    ephemeral_forced?: boolean;
    kg_aware_retrieval_enabled?: boolean;
  };
  restart_required?: boolean;
}

export interface KGRetrievalDiagnostics {
  requested: boolean;
  effective: boolean;
  attempted: boolean;
  applied: boolean;
  snapshot_version_id?: string | null;
  wiki_snapshot_id?: string | null;
  candidate_count: number;
  hydrated_count: number;
  adopted_evidence_ids: string[];
  reason_codes: string[];
}

export interface Citation {
  id: string;
  document_id: string;
  url: string | null;
  title: string;
  snippet: string;
  score: number;
  origin?: EvidenceOrigin;
  artifact_type?: EvidenceArtifactType;
  artifact_id?: string;
  grounding_eligible?: boolean;
  provenance_refs?: string[];
  source_status?: EvidenceSourceStatus;
  published_at?: number | null;
  updated_at?: number | null;
  effective_at?: number | null;
  retrieved_at?: number | null;
  search_provider?: SearchProviderId | null;
  fetch_provider?: Exclude<SearchProviderId, "brave"> | null;
}

export interface EvidenceConflict {
  id: string;
  claim_id: string;
  claim: string;
  type: "temporal" | "scope_entity" | "authority_directness" | "internal_public_scope" | "peer_unresolved";
  claim_scope: string;
  evidence_ids: string[];
  resolution: "contested";
  reason: string;
}

export type ConfidenceLabel = "established" | "contested" | "uncertain";

export interface GroundedAnswer {
  text: string;
  citations: Citation[];
  claim_map: Record<string, string[]>;
  coverage: number;
  summary?: string;
  key_takeaways?: string[];
  limitations?: string;
  confidence_label?: ConfidenceLabel | string;
  conflicts?: EvidenceConflict[];
}

export interface EvidenceSource {
  chunk_id: number;
  document_id: string;
  url: string | null;
  title: string;
  text: string;
  score: number;
  channels?: string[];
  origin?: EvidenceOrigin;
  artifact_type?: EvidenceArtifactType;
  artifact_id?: string;
  grounding_eligible?: boolean;
  provenance_refs?: string[];
  source_status?: EvidenceSourceStatus;
  published_at?: number | null;
  updated_at?: number | null;
  effective_at?: number | null;
  retrieved_at?: number | null;
  search_provider?: SearchProviderId | null;
  fetch_provider?: Exclude<SearchProviderId, "brave"> | null;
}

export interface RetrievalOutcome {
  lane_id: string;
  lane: EvidenceOrigin;
  query: string;
  attempted: boolean;
  status: "completed" | "empty" | "failed" | "timeout" | "cancelled";
  provider?: string;
  search_provider?: SearchProviderId;
  fetch_providers?: Exclude<SearchProviderId, "brave">[];
  providers_attempted?: SearchProviderId[];
  providers_skipped?: SearchProviderId[];
  provider_failure_reasons?: Partial<Record<SearchProviderId, string>>;
  tier?: string;
  hit_count: number;
  evidence_ids: string[];
  error_code?: string;
  reason?: string;
  locale?: RetrievalLocaleOutcome;
  search_options?: RetrievalSearchOptionsOutcome;
  started_at: number;
  finished_at: number;
}

export interface TemporalCoverage {
  intent: "none" | "latest" | "as_of";
  requested_as_of?: number;
  checked_at: number;
  status: "not_requested" | "bounded" | "partially_bounded" | "unknown";
  supported_through?: number;
  basis: "claim_effective_at" | "source_published_or_updated_at" | "none";
  evidence_ids: string[];
  reason_codes: string[];
  statement: string;
  note_required: boolean;
}

export interface ResearchIntent {
  original_query: string;
  research_focus: string;
  source_requirement: "none" | "authoritative" | "primary";
  citation_limit?: number | null;
  output_shape: "default" | "concise" | "sentence_count" | "max_length";
  sentence_count?: number | null;
  max_length?: number | null;
  parse_status: "parsed" | "partial" | "unchanged";
  reason_codes: string[];
}

export interface EvidenceEligibilityOutcome {
  evidence_id: string;
  decision: "eligible" | "rejected" | "unknown";
  relevance: "strong" | "weak" | "unrelated" | "unknown";
  authority_status: "not_requested" | "satisfied" | "unsatisfied" | "unknown";
  reason_codes: string[];
  matched_anchors: string[];
  provenance_family: string;
}

export interface RemediationOutcome {
  attempted: boolean;
  attempt_index: 0 | 1;
  trigger?: string | null;
  action?: "retrieve_and_rewrite" | "rewrite_only" | null;
  focused_query: string;
  excluded_evidence_ids: string[];
  excluded_provenance_roots: string[];
  lane_ids: string[];
  new_evidence_ids: string[];
  status: "not_attempted" | "completed" | "exhausted" | "failed" | "timeout" | "cancelled";
  started_at?: number | null;
  finished_at?: number | null;
  source_generation_id?: string | null;
  replacement_generation_id?: string | null;
}

export interface CouncilMember {
  model: string;
  answer: GroundedAnswer;
  perspective?: string;
}

export interface CouncilMemberFailure {
  model: string;
  code: "MEMBER_EXECUTION_FAILED";
}

export interface EvidenceMatrixRow {
  subtask: string;
  queries: string[];
  notes: string;
  gaps: string[];
  source_count: number;
  status: string;
}

export interface ResearchState {
  plan: string[];
  evidence_matrix: EvidenceMatrixRow[];
  searches_used: number;
  searches_max: number;
  pages_fetched: number;
  think_tokens_used: number;
  gap_rounds: number;
}

export type DeepCheckpointStage = "plan_ready" | "round_complete" | "final_evidence_ready";
export type DeepCheckpointReadiness = "unavailable" | "ready" | "degraded" | "blocked";

export interface DeepCheckpointSettings {
  auto_checkpoint_enabled: boolean;
}

export interface DeepCheckpointSummary {
  run_id: string;
  lineage_id?: string;
  revision: number;
  schema_version: number;
  stage: DeepCheckpointStage;
  disposition: "ready" | "claimed" | "abandoned" | string;
  claimed_child_run_id?: string | null;
  created_at?: number;
  updated_at?: number;
  integrity: "absent" | "valid" | "invalid";
  resume_readiness: DeepCheckpointReadiness;
  reason_codes: string[];
  budget_usage?: {
    searches_used?: number;
    searches_max?: number;
    pages_fetched?: number;
    gap_rounds?: number;
    recovery_searches_used?: number;
    recovery_searches_max?: number;
  };
}

export interface DeepCheckpointContinueRequest {
  expected_checkpoint_revision: number;
  idempotency_key: string;
  action: "continue" | "revise";
  incremental_budget?: Record<string, number>;
  remaining_plan?: { sections: string[] };
}

export interface RunResult {
  run_id: string;
  mode: Mode;
  status: string;
  answer: GroundedAnswer;
  sources?: EvidenceSource[];
  lanes?: { query: string; sources: EvidenceSource[] }[];
  sections?: { title: string; answer: GroundedAnswer }[];
  members?: CouncilMember[];
  member_failures?: CouncilMemberFailure[];
  confidence?: number;
  breakdown?: Record<string, number>;
  consensus_strength?: string;
  gap_rounds?: number;
  completion_reason?: "timeout" | string;
  budget_exhausted?: boolean;
  research_state?: ResearchState;
  evidence_matrix?: EvidenceMatrixRow[];
  retrieval_outcomes?: RetrievalOutcome[];
  degraded_reasons?: string[];
  unverified_items?: UnverifiedItem[];
  verification?: ReportVerification;
  confidence_warning?: ConfidenceWarning;
  temporal_coverage?: TemporalCoverage;
  conflicts?: EvidenceConflict[];
  stream_metrics?: StreamMetrics;
  writer_diagnostics?: WriterDiagnostics;
  research_intent?: ResearchIntent;
  evidence_eligibility?: EvidenceEligibilityOutcome[];
  remediation?: RemediationOutcome;
  recovery_searches_max?: number;
  recovery_searches_used?: number;
  phase_metrics?: PhaseMetric[];
  deadline_summary?: DeadlineSummary;
  retrieval_budget?: RetrievalBudget;
  resolved_search_usage?: ResolvedSearchUsage;
  kg_retrieval?: KGRetrievalDiagnostics;
}

export type ReportVerificationStatus = "pending" | "running" | "completed" | "failed";

export interface ReportVerification {
  status: ReportVerificationStatus;
  started_at?: number;
  completed_at?: number;
  reason_codes?: string[];
}

export interface ConfidenceWarning {
  reasons: string[];
  acknowledged: boolean;
  acknowledged_at?: number;
}

export type UnverifiedReason =
  | "evidence_insufficient"
  | "verification_incomplete"
  | "source_conflict"
  | "unsupported_claim";

export interface UnverifiedItem {
  question: string;
  reason: UnverifiedReason;
}

export interface BudgetSnapshot {
  mode: Mode | string;
  searches_used: number;
  searches_max: number;
  stage: string;
  gap_rounds?: number;
  pages_fetched?: number;
  remaining_searches?: number;
  search_tier?: "brave" | "firecrawl_search" | "cache" | "indexed_external" | "snippet_only" | string;
  degraded_fetch?: boolean;
  recovery_searches_max?: number;
  recovery_searches_used?: number;
}

export interface Run {
  id: string;
  session_id: string;
  mode: Mode;
  query: string;
  status: string;
  budget?: Record<string, number>;
  result?: RunResult;
  error?: { error?: { code: string; message: string } };
  citations?: Citation[];
  metadata?: {
    search_options?: SearchOptions;
    requested_locale?: SearchLocaleSnapshot;
    effective_locale?: SearchLocaleSnapshot;
    requested_at?: number;
    stream_metrics?: StreamMetrics;
    writer_diagnostics?: WriterDiagnostics;
    phase_metrics?: PhaseMetric[];
    deadline_summary?: DeadlineSummary;
    retrieval_budget?: RetrievalBudget;
    resolved_search_usage?: ResolvedSearchUsage;
    auto_checkpoint_enabled?: boolean;
    kg_aware_retrieval_enabled?: boolean;
    kg_retrieval_snapshot?: Record<string, unknown>;
    kg_retrieval?: KGRetrievalDiagnostics;
    checkpoint_parent_run_id?: string;
    checkpoint_revision?: number;
    lineage_id?: string;
    checkpoint_action?: "continue" | "revise";
    checkpoint_resume_readiness?: DeepCheckpointReadiness;
  };
  created_at?: number;
  updated_at?: number;
}

export interface RunEvent {
  seq: number;
  type: string;
  payload: Record<string, unknown>;
}

export interface ProviderStatus {
  llm_base_url: string;
  llm_model: string;
  has_llm_key: boolean;
  brave_ready: boolean;
  firecrawl_ready: boolean;
  exa_ready?: boolean;
  tavily_ready?: boolean;
  search_providers?: Record<SearchProviderId, SearchProviderCapability>;
  provider_config_revision?: number;
  external_search_service?: ExternalSearchServiceStatus;
  embed_model: string;
  ollama_base_url: string;
}

export interface ProviderUpdate {
  llm_base_url?: string;
  llm_api_key?: string;
  llm_model?: string;
  brave_api_key?: string;
  firecrawl_api_key?: string;
  firecrawl_base_url?: string;
  exa_api_key?: string;
  tavily_api_key?: string;
}

export type ModelKind = "chat" | "embed";

export interface ModelCapabilities {
  vision: boolean;
  ocr: boolean;
  web_search: boolean;
  thinking: boolean;
}

/** Coarse user-facing thinking dial; maps to a token budget on the backend. */
export type ThinkingEffort = "off" | "low" | "medium" | "high";

export interface ModelProfile {
  id: string;
  alias: string;
  kind: ModelKind;
  context_window: number;
  capabilities: ModelCapabilities;
  thinking_budget: number | null;
  temperature: number | null;
  dimensions: number | null;
  enabled: boolean;
}

export interface LLMProviderProfile {
  id: string;
  label: string;
  type: string;
  base_url: string;
  models: ModelProfile[];
  enabled: boolean;
  has_key: boolean;
}

export interface LLMRoles {
  default: string;
  embed: string;
  memory: string;
  council: string[];
}

export interface LLMRegistryStatus {
  providers: LLMProviderProfile[];
  roles: LLMRoles;
}

export interface ModelProfileInput {
  id: string;
  alias?: string;
  kind?: ModelKind;
  context_window?: number;
  capabilities?: Partial<ModelCapabilities>;
  thinking_budget?: number | null;
  temperature?: number | null;
  dimensions?: number | null;
  enabled?: boolean;
}

export interface LLMProviderUpsert {
  id: string;
  label?: string;
  type?: string;
  base_url?: string;
  api_key?: string;
  models?: (string | ModelProfileInput)[];
  enabled?: boolean;
}

export interface ProviderCandidate {
  provider_id?: string;
  type?: string;
  base_url?: string;
  api_key?: string;
  model?: string;
  /** Probe mode: omit for provider-level GET models; "chat" for a model-level chat test. */
  probe?: "chat";
}

export interface ModelListResult {
  ok: boolean;
  status: string;
  models: string[];
  latency_ms: number | null;
  endpoint_host: string;
  error_message: string;
}

export interface LLMTestResult {
  ok: boolean;
  status: string;
  latency_ms: number | null;
  model?: string;
  endpoint_host: string;
  missing_required?: string[];
  error_message: string;
  /** Catalog size from a provider-level GET models probe (null for chat probes). */
  models_count?: number | null;
}

export interface ToolsTestResult {
  ok: boolean;
  status: string;
  latency_ms: number | null;
  provider?: SearchProviderId;
  capabilities?: { search: boolean; extract: boolean };
  endpoint_host?: string;
  quick_answer_ready?: boolean;
  report_ready?: boolean;
  brave_status?: string;
  firecrawl_status?: string;
  brave_host?: string;
  firecrawl_host?: string;
  missing_required?: string[];
  error_message: string;
  probe_state?: SearchProviderCapability["probe_state"];
  credential_state?: SearchProviderCapability["credential_state"];
  availability?: SearchProviderCapability["availability"];
  retry_after?: number | null;
  reason_codes?: string[];
}

export interface Session {
  id: string;
  title: string;
  updated_at: number;
  pinned?: boolean;
  mode_default?: Mode | null;
  search_usage_policy?: SearchUsagePolicy | null;
  latest_run_id?: string | null;
  latest_run_mode?: Mode | null;
  latest_run_status?: string | null;
  resumable_checkpoint?: DeepCheckpointSummary | null;
}

export interface ArchivedSession {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  archived_at: number;
  pinned?: boolean;
  mode_default?: Mode | null;
  search_usage_policy?: SearchUsagePolicy | null;
  latest_run_mode?: Mode | null;
}

export interface Message {
  id: string;
  session_id: string;
  role: string;
  content: string;
  created_at: number;
  score?: number;
  /** JSON-encoded turn metadata (e.g. {"run_id": "...", "mode": "fast"}). */
  metadata?: string | null;
}

export interface MemoryItem {
  id: string;
  layer: string;
  namespace?: string;
  scope_key?: string;
  content: string;
  salience: number;
  effective_salience: number;
  pinned?: boolean;
  expires_at?: number | null;
  archived_at?: number | null;
  metadata?: Record<string, unknown>;
}

export interface MemoryBatchExpiryResult {
  namespace: "workspace" | "user";
  dry_run: boolean;
  would_update: number;
  updated: number;
  skipped_pinned: number;
  skipped_archived: number;
  failed: number;
}

export interface LibraryIndexWarning {
  code: string;
  message: string;
}

export interface MemoryOptimizationGroup {
  group_id: string;
  similarity: number;
  items: MemoryItem[];
  suggested_summary: string;
  suggestions: { type: "merge" | "archive" | "downrank"; memory_ids: string[]; keep_id?: string; factor?: number }[];
}

export interface MemoryOptimizationPreview {
  groups: MemoryOptimizationGroup[];
  threshold: number;
}

export interface MemoryDreamStatus {
  status: "idle" | "queued" | "running" | "cancelling" | "cancelled" | "completed" | "error";
  dream_run_id?: string | null;
  workspace_key: string;
  user_key: string;
  started_at?: number | null;
  finished_at?: number | null;
  updated_at?: number | null;
  cancel_requested_at?: number | null;
  backend?: "thread" | "rq" | "sync" | string | null;
  backend_job_id?: string | null;
  memories_created: number;
  error?: string | null;
  error_code?: string | null;
  failed_stage?: string | null;
  failed_session_id?: string | null;
  attempted_sessions?: number;
  successful_sessions?: number;
  failed_sessions?: number;
  user_profile_attempted?: boolean;
  user_profile_updated?: boolean;
  user_profile_failed?: boolean;
  failure_count?: number;
  provider_ref?: string | null;
  structured_strategy?: string | null;
  response_shape?: string | null;
  finish_reason?: string | null;
  validation_error?: string | null;
  incomplete?: boolean;
  incomplete_reasons?: string[];
  workspace_memory_refresh_strategy?: WorkspaceMemoryRefreshStrategy;
  workspace_skipped?: number;
  metadata?: Record<string, unknown>;
  schedule?: MemoryDreamSchedule;
  next_scheduled_at?: number | null;
  schedule_due?: boolean;
}

export interface SessionMemoryGroup {
  session_id: string;
  session_title: string;
  session_updated_at?: number | null;
  session_archived_at?: number | null;
  memories: MemoryItem[];
}

export interface SessionMemoryList {
  sessions: SessionMemoryGroup[];
  total: number;
  limit: number;
  offset: number;
  truncated: boolean;
}

export type MemoryDreamSchedule = "off" | "daily" | "weekly";
export type WorkspaceMemoryRefreshStrategy = "incremental" | "full";

export interface MemoryProbeDiagnostics {
  response_shape?: string | null;
  finish_reason?: string | null;
  validation_error?: string | null;
  response_empty?: boolean | null;
  attempts?: number | { ok?: boolean; structured_strategy?: string; error_code?: string; response_shape?: string }[];
}

export interface MemoryFormatProbeResult {
  status: "passed" | "caution" | "failed";
  provider_ref: string;
  capability: string;
  structured_strategy: string;
  checked_at: number;
  error_code?: string;
  diagnostics: MemoryProbeDiagnostics;
  stale?: boolean;
  stale_reason?: string;
}

export interface MemorySettings {
  post_turn_memory_refresh_enabled: boolean;
  workspace_memory_refresh_strategy: WorkspaceMemoryRefreshStrategy;
  memory_thinking_enabled: boolean;
  format_probe: MemoryFormatProbeResult | null;
}

export interface MemoryWriteAudit {
  status: string;
  source: string;
  workspace_key: string;
  user_key: string;
  attempted: number;
  succeeded: number;
  skipped: number;
  failed: number;
  error_code?: string | null;
  checked_at?: number | null;
  provider_ref?: string | null;
  stale?: boolean;
  incomplete_reasons?: string[];
}

export interface PostRunNotesConfig {
  enabled_modes: Mode[];
}

export interface LibraryDocument {
  id: string;
  url?: string | null;
  title: string;
  content_type: string;
  created_at: number;
  chunks: number;
  origin?: "external" | "local";
  artifact_type?: "web_document" | "library_document";
  artifact_id?: string;
  filename?: string | null;
  deletable?: boolean;
  content_sha256?: string | null;
}

export interface LocalDocumentImportResponse {
  status: "created" | "existing";
  document: {
    id: string;
    title: string;
    filename: string;
    content_type: "pdf" | "docx" | "xlsx";
    origin: "local";
    artifact_type: "library_document";
    content_sha256: string;
    chunk_count: number;
  };
  warnings: string[];
}

export interface LibraryDocumentDeleteResponse {
  status: "deleted" | "deleted_with_cleanup_warning";
  document_id: string;
  origin: "external" | "local";
  artifact_type: "web_document" | "library_document";
  chunks_deleted: number;
  snapshots_deleted: number;
  provenance_deleted: number;
  warnings: string[];
}

export interface Note {
  id: string;
  content: string;
  created_at?: number;
  session_id?: string | null;
  run_id?: string | null;
  metadata?: Record<string, unknown>;
}

export interface LibraryProvenance {
  id?: string;
  artifact_type?: string;
  artifact_id?: string;
  session_id?: string | null;
  run_id?: string | null;
  document_id?: string | null;
  url?: string | null;
  created_at?: number;
  metadata?: Record<string, unknown>;
}

export interface LibraryReport {
  id: string;
  title: string;
  body_md?: string;
  body_preview?: string;
  mode: string;
  parent_report_id?: string | null;
  session_id?: string | null;
  run_id?: string | null;
  created_at: number;
  metadata?: Record<string, unknown>;
  children?: LibraryReport[];
}

export interface PageSnapshot {
  id: string;
  document_id?: string | null;
  url: string;
  captured_at?: number;
  mime_type?: string;
  size_bytes?: number;
  status: string;
  error?: string | null;
  text?: string;
}

/** Effective context-injection budget (DR-ITER-057); mirrors the backend to_dict. */
export interface ContextBudgetConfig {
  preset: string;
  max_chars: number;
  turns: number;
  recent_chars: number;
  semantic_chars: number;
  memory_chars: number;
  recent_share: number;
  semantic_share: number;
  memory_share: number;
}

export interface ContextBudgetState {
  effective: ContextBudgetConfig;
  saved: ContextBudgetConfig | null;
  env: ContextBudgetConfig | null;
  default: ContextBudgetConfig;
  source: "saved" | "env" | "default";
}

export interface ContextBudgetRecommendation {
  recommendation: ContextBudgetConfig;
  reasons: string[];
}

export interface KnowledgeGraphSettings {
  post_run_rebuild_enabled: boolean;
  wiki_auto_refresh_after_kg_rebuild: boolean;
}

export type KnowledgeGraphNodeType = "session" | "source" | "note" | "report" | "memory" | string;

export interface KnowledgeGraphSessionRef {
  node_key: string;
  session_id: string;
  title: string;
  summary: string;
  jump_uri?: string;
  assistant_message_id?: string;
  assistant_created_at?: string;
}

export interface KnowledgeGraphVersion {
  id: string;
  workspace_id: string;
  status: string;
  is_active: boolean;
  started_at?: number | null;
  finished_at?: number | null;
  activated_at?: number | null;
  stats_json?: Record<string, unknown>;
}

export interface KnowledgeGraphBuildJob {
  id: string;
  version_id?: string | null;
  workspace_id: string;
  backend: string;
  backend_job_id?: string | null;
  status: "queued" | "running" | "succeeded" | "partial" | "failed" | string;
  stage: string;
  progress: number;
  started_at?: number | null;
  finished_at?: number | null;
  error_code?: string | null;
  error_message?: string | null;
  stats_json?: Record<string, unknown>;
}

export interface KnowledgeGraphJobEventPayload {
  job_id: string;
  job_kind: "graph" | "wiki";
  status: "queued" | "running" | "succeeded" | "partial" | "failed" | string;
  stage: string;
  progress?: number;
  emitted_at?: number;
  attempted_sessions?: number;
  succeeded_sessions?: number;
  skipped_sessions?: number;
  failed_sessions?: number;
  error_code?: string;
  stats?: Record<string, unknown>;
}

export interface KnowledgeGraphJobEvent {
  seq: number;
  type: "job_progress" | "job_terminal" | string;
  payload: KnowledgeGraphJobEventPayload;
}

export interface KnowledgeGraphNode {
  id: string;
  type: KnowledgeGraphNodeType;
  key: string;
  title: string;
  summary: string;
  attrs: Record<string, unknown>;
  confidence: number;
  visible: boolean;
  referencing_sessions?: KnowledgeGraphSessionRef[];
}

export interface KnowledgeGraphEdge {
  id: string;
  type: string;
  source: string;
  target: string;
  weight: number;
  confidence: number;
  visible: boolean;
  attrs: Record<string, unknown>;
}

export interface KnowledgeGraphProvenance {
  id: string;
  owner_kind: string;
  source_kind: string;
  source_id: string;
  span: Record<string, unknown>;
  evidence: Record<string, unknown>;
  created_at?: number | null;
}

export interface KnowledgeGraphStatus {
  workspace_id: string;
  active_version: KnowledgeGraphVersion | null;
  last_build_job: KnowledgeGraphBuildJob | null;
  counts: { nodes: number; edges: number };
}

export interface KnowledgeGraphGraph {
  workspace_id: string;
  active_version: KnowledgeGraphVersion | null;
  nodes: KnowledgeGraphNode[];
  edges: KnowledgeGraphEdge[];
}

export interface KnowledgeGraphNodeSet {
  count: number;
  node_keys: string[];
  members: KnowledgeGraphNode[];
}

export interface KnowledgeGraphSessionLane {
  session: KnowledgeGraphNode & {
    assistant_summary?: string;
    assistant_excerpt?: string;
    last_assistant_message_id?: string;
    last_assistant_at?: string;
  };
  reports: KnowledgeGraphNode[];
  notes: KnowledgeGraphNode[];
  source_set: KnowledgeGraphNodeSet;
  memory_set: KnowledgeGraphNodeSet;
  counts: { reports: number; notes: number; sources: number; memories: number };
}

export interface KnowledgeGraphSessionMap {
  workspace_id: string;
  active_version: KnowledgeGraphVersion | null;
  layout_mode: "session_tree";
  session_limit: number;
  member_preview: number;
  sessions: KnowledgeGraphSessionLane[];
  nodes_by_key: Record<string, KnowledgeGraphNode>;
  edges: KnowledgeGraphEdge[];
}

export interface KnowledgeGraphWikiSnapshot {
  id: string;
  workspace_id: string;
  graph_version_id: string;
  status: "succeeded" | "partial" | "failed" | string;
  is_active: boolean;
  stale: boolean;
  stale_reason: string;
  started_at?: number | null;
  finished_at?: number | null;
  activated_at?: number | null;
  stats: KnowledgeGraphWikiStats;
  error_code?: string;
  error_message?: string;
}

export interface KnowledgeGraphWikiTopic {
  id: string;
  snapshot_id: string;
  key: string;
  title: string;
  summary: string;
  depth: number;
  parent_id: string;
  session_ids: string[];
  source_message_ids: string[];
  attrs: Record<string, unknown>;
}

export interface KnowledgeGraphWikiEvidenceRef {
  type?: string;
  id?: string;
  node_key?: string;
  excerpt?: string;
}

export interface KnowledgeGraphWikiStats extends Record<string, unknown> {
  eligible_sessions?: number;
  fresh_sessions?: number;
  carried_sessions?: number;
  uncovered_sessions?: number;
  candidate_coverage?: number;
  active_coverage?: number;
  activation_decision?: "activated" | "retained_previous" | string;
  activation_reason?: string;
  topic_count?: number;
  top_level_topic_count?: number;
  max_depth?: number;
  topics_with_evidence?: number;
  evidence_coverage?: number;
  question_like_title_count?: number;
  invalid_summary_count?: number;
  orphan_root_count?: number;
}

export interface KnowledgeGraphWikiEdge {
  id: string;
  snapshot_id: string;
  source: string;
  target: string;
  type: string;
  attrs: Record<string, unknown>;
}

export interface KnowledgeGraphWikiCrossLink extends KnowledgeGraphWikiEdge {
  evidence_type: string;
  evidence_refs: Array<Record<string, unknown>>;
}

export interface KnowledgeGraphWikiRefreshJob {
  id: string;
  snapshot_id: string;
  graph_version_id: string;
  workspace_id: string;
  backend: string;
  backend_job_id: string;
  status: "queued" | "running" | "succeeded" | "partial" | "failed" | string;
  stage: string;
  progress: number;
  started_at?: number | null;
  finished_at?: number | null;
  attempted_sessions: number;
  succeeded_sessions: number;
  skipped_sessions: number;
  failed_sessions: number;
  error_code?: string;
  error_message?: string;
  stats: KnowledgeGraphWikiStats;
}

export interface KnowledgeGraphWikiMap {
  workspace_id: string;
  layout_mode: "wiki_map";
  active_graph_version: KnowledgeGraphVersion | null;
  active_snapshot: KnowledgeGraphWikiSnapshot | null;
  stale: boolean;
  stale_reason: string;
  stats: KnowledgeGraphWikiStats & { topics?: number; tree_edges?: number; cross_links?: number };
  topics: KnowledgeGraphWikiTopic[];
  tree_edges: KnowledgeGraphWikiEdge[];
  cross_links: KnowledgeGraphWikiCrossLink[];
  last_refresh_job: KnowledgeGraphWikiRefreshJob | null;
}

export interface KnowledgeGraphNodeDetail extends KnowledgeGraphNode {
  edges: KnowledgeGraphEdge[];
  provenance: KnowledgeGraphProvenance[];
}
