# Deep Research Studio Web API

This document is the stable contract for the local-first baseline API. It is
derived from `deepresearch_core/baseline/api.py` and is checked by
`tests/baseline/test_api_spec_drift.py`.

## Runtime

- Prefix: `/api/v1`
- Auth: `DEEPRESEARCH_AUTH_MODE=off|token`; `/api/v1/health` stays public.
- CORS is disabled by default. The packaged Web UI uses the same-origin API
  proxy; wildcard origins and credentialed CORS are forbidden.
- Error envelope: `{"error":{"code":"...","message":"...","details":{...}}}`.
- Health fields include `job_backend`, `redis_reachable`, `db_encrypted`,
  `auth_mode`, `vector_backend`, `post_run_notes_modes`,
  `chroma_persist_enabled`, `chroma_persist_mode`, and a secret-free
  `chroma_persist_path`.
- Memory edits use `PATCH /api/v1/memories/{memory_id}` for workspace/user
  memories only. The body accepts `content`, `salience`, `metadata`, `pinned`,
  and/or `expires_at` (`null` clears expiry); archived, session-layer, unknown,
  or unsupported-field requests return `INVALID_REQUEST`. Setting a TTL on a
  pinned memory returns `409 MEMORY_PINNED`; pinning atomically clears
  `expires_at`. Pinned memories are
  excluded from optimizer preview and rejected by optimizer apply actions
  (`archive`, `downrank`, `merge`). `library_index_warning` remains in the
  response shape for compatibility; memory-only success returns `null` because
  local memories are no longer re-indexed into Library.
- `GET /api/v1/memories/{memory_id}` returns one active or archived Memory row
  for citation deep links. `POST /api/v1/memories/cleanup` performs a bounded,
  idempotent expiry sweep (`limit` 1–500). The body either selects one scope
  with `namespace=workspace|user` plus `scope_key`, or uses the compatible
  `workspace_key` / `user_key` pair; the two targeting forms cannot be mixed
  and unknown fields return `INVALID_REQUEST`. The response returns
  `checked_at`, `scanned`, `archived`, `memory_ids[]`, and an exact cross-scope
  `remaining` count. Expired, unpinned rows are archived, never deleted; pinned
  rows are not changed.
- Memory batch expiry uses `POST /api/v1/memories/batch-expiry` for workspace/user
  memories only. The body accepts `namespace` (`workspace|user`), `workspace_key`,
  `user_key`, `expires_at` (`null` clears expiry), and `dry_run`. The response is
  secret-free and includes `namespace`, `dry_run`, `would_update`, `updated`,
  `skipped_pinned`, `skipped_archived`, and `failed`. Pinned and archived memories
  are never modified; archived rows are still counted as `skipped_archived`.
- Session preferences use `PATCH /api/v1/sessions/{session_id}` for local UX
  fields `pinned` and `mode_default`. Active session lists sort pinned sessions
  first, then by `updated_at DESC`. Run creation still accepts an explicit
  `mode`; if a client omits `mode` and supplies a session with `mode_default`,
  the service fills that default before validating the run payload.
- `GET /api/v1/library/documents` is the Library Sources surface and returns
  user-visible external `http(s)` sources plus imported local documents whose
  metadata identifies `origin=local`, `artifact_type=library_document`, and
  `source=local_import`. Legacy Memory/Note/Report-derived documents and hidden
  retrieval candidates remain excluded.
- `POST /api/v1/library/documents/import` accepts one bounded multipart PDF,
  DOCX, or XLSX file plus an optional title. The service verifies extension,
  MIME, signature, and OOXML package structure; parses in an isolated,
  time-bounded subprocess; stores extracted text only; and commits through the
  lexical index with embeddings disabled. The 10 MiB limit is enforced while
  reading the upload and does not trust `Content-Length`. Identical bytes return
  `status=existing` with the original document id and title.
- `DELETE /api/v1/library/documents/{document_id}` deletes only a user-visible
  external Source or a T56 local import. It removes current chunks, FTS rows,
  snapshots, and provenance while preserving historical citations, runs,
  reports, and graph versions. Cleanup failures return
  `deleted_with_cleanup_warning`; Note, Report, Memory, and hidden-candidate
  documents return `DOCUMENT_NOT_DELETABLE`.
- Cache purge uses `DELETE /api/v1/cache?scope=all|search|fetch`; it removes
  only retrieval cache rows and never deletes sessions, runs, memories, Library
  data, or BYOK settings.
- Run cost preview uses `POST /api/v1/runs/estimate-cost`. It accepts a draft
  run payload with `mode`, optional `budget`, `llm_model`, `council_models`,
  `council_size`, `council_strategy`, and `search_options`; it intentionally
  does not require `query` and never creates a run, calls a provider, or writes
  data. The response is secret-free and includes estimated search count,
  optional Brave USD search cost, optional Firecrawl search credits, estimated
  LLM input/output tokens, optional LLM/total USD ranges, `confidence`,
  `pricing_basis`, and disclaimers. Unknown or local model pricing is returned
  as `null` rather than guessed.
- Council run results may include optional `member_failures[]`. Each item is
  exactly a bounded `model` label plus `code="MEMBER_EXECUTION_FAILED"`; no raw
  exception or provider response is exposed. One failed member does not abort
  the remaining drafts. If every member fails, the run returns a redacted
  `UPSTREAM_ERROR` without invoking the judge/validator.
- Citation grounding splits English and CJK terminal punctuation even without
  whitespace and uses Latin/digit terms plus CJK bigrams. A response with no
  substantive claim has coverage `0.0`, not an implicit full-coverage pass.
- Fast, Multiple, and Deep runs plan independent `local`, `memory`, and
  `external` evidence lanes and execute them in the fixed adapter-safe order
  `local -> memory -> external`; an empty, failed, or timed-out lane never
  triggers or suppresses another lane. Run results may include
  `retrieval_outcomes[]`, `degraded_reasons[]`, and `temporal_coverage`.
  `latest/current/today` and explicit `as_of` intent receive one supplemental
  three-lane temporal check. Missing fact-time coverage does not raise
  `GROUNDING_REQUIRED`; it returns `completed_degraded` with a deterministic
  temporal statement/note. Zero citable evidence and final citation coverage
  below the grounding threshold remain fail-closed.
- Run creation freezes `requested_at` plus absolute `deadline_at` before enqueue;
  RQ queue delay consumes the same budget. A worker that starts after expiry
  returns `RUN_TIMEOUT` without provider construction. Mode-specific internal
  retrieval budgets cap candidates, normal/recovery fetches, fetch concurrency,
  and writer/finalize/cleanup reserves; a requested timeout below the reserve
  total is `INVALID_REQUEST`.
- External retrieval is two-stage: immutable candidates and conservatively
  qualified snippets first, then only the pages needed for readiness are fetched.
  Redirect/canonical mismatch, truncation, missing identity, and `unknown`
  candidates cannot become final evidence without requalification. Pure network
  work may overlap local/memory; all SQLite/cache/index writes and merges remain
  deterministic on the orchestrator thread.
- Every terminal run may add `phase_metrics[]`, `deadline_summary`, and
  `retrieval_budget`; stream metrics may add `pre_writer_ms` and
  `provider_first_delta_ms`. Missing fields on historical runs are valid. Result,
  terminal status, and citation rows commit atomically. `answer_reset.reason`
  includes `deadline_exceeded`; only an already grounded result may become
  `partial_timeout`.
- Run-result citations add `origin=external|local|memory`, `artifact_type`,
  `artifact_id`, `grounding_eligible`, `provenance_refs[]`, `source_status`, and
  separate `published_at`, `updated_at`, `effective_at`, and `retrieved_at`
  fields. Retrieval time is never treated as the fact's effective time. The
  current runtime always supplies `retrieved_at`; historical serialized results
  and legacy citation rows may omit it and are read using their persisted
  creation time as the compatibility fallback. Cross-lane fusion uses rank/RRF
  plus provenance-family deduplication instead of comparing provider-specific
  raw scores, and retains lane coverage whenever the result capacity permits.
  `artifact_type=library_document` identifies an imported local document;
  its public URL is empty, its stable key is
  `local:library_document:{document_id}`, and its UI deep link is
  `/library/sources/{document_id}`.
  answer may include unresolved `conflicts[]`; only genuine unresolved peer
  conflicts set `confidence_label=contested` and produce a post-body conflict
  note. Deterministic conflict candidates are built from the complete normalized
  evidence pack before composition, so omitting one side from `claim_map` cannot
  suppress analysis. `origin`, result counts, and same-root repost counts are
  not tie-breakers.
- Query-based Memory recall applies `min_similarity=0.35` before salience fusion
  when a valid query embedding and comparable candidate embedding exist. If the
  query embedder is missing, fails, or returns an invalid vector, the existing
  salience-only fallback remains in effect.
- Retrieval settings use `GET/PATCH /api/v1/settings/retrieval` and persist
  `retrieval.chroma_persist_enabled` in `config.json` while preserving other
  config blocks. New local SQLite-backed runtimes default this value to `true`;
  `DEEPRESEARCH_CHROMA_EPHEMERAL=1` and `:memory:` databases force effective
  ephemeral mode. PATCH returns `restart_required: true`; Chroma remains a
  derived ANN cache, while `chunks.embedding` in SQLite is the only embedding
  source of truth.
- Search usage settings use `GET/PATCH /api/v1/settings/search-usage`. PATCH
  accepts one complete normalized `single`, `dual_role`, or `auto` policy and
  persists it independently from BYOK credentials. Session and run payloads may
  carry `search_usage_policy` as an object, `null`, or omit the field; the
  effective search/extract chains are frozen before enqueue and remain stable
  for that run.
- External Search Provider credentials use
  `POST /api/v1/settings/search-providers/save` with `expected_revision` plus per-provider
  drafts. The service probes enabled credentials, accepts eligible drafts per
  provider, and commits all accepted items once under CAS; `success`, `partial`,
  `rejected`, and `noop` describe the operation, not overall availability.
  `PATCH /api/v1/settings/search-providers/enabled` atomically toggles one provider
  for future runs. `POST /api/v1/tools/test` remains transient. env/secure-owned
  credentials are read-only and never copied into plaintext config.
- Provider readiness covers only live Brave/Tavily/Exa/Firecrawl calls.
  cache/indexed_external retain external provenance but do not make a live
  Provider ready; local/Library and Memory evidence lanes are never disabled by
  External search state. External unavailable warns but does not prevent run
  creation; grounding remains authoritative.
- Knowledge Graph settings use `GET/PUT /api/v1/settings/knowledge-graph` and
  expose `post_run_rebuild_enabled` plus
  `wiki_auto_refresh_after_kg_rebuild` (default `false`). Disabling post-run
  rebuild stops automatic graph rebuilds, but manual
  `POST /api/v1/knowledge-graph/rebuild` remains available. Enabling Wiki
  auto-refresh starts a Wiki Map refresh only after a graph rebuild successfully
  commits a new active version.
- Session Wiki Map uses `GET /api/v1/knowledge-graph/session-map` as an
  additive view model over the active graph version. Query parameters are
  `workspace_id`, `session_limit`, and `member_preview` (`0` means return hub
  counts and node keys without preview inflation; nonzero returns bounded
  lightweight member payloads through the same view model). The response includes
  `workspace_id`, `active_version`, `layout_mode=session_tree`, `session_limit`,
  `member_preview`, `sessions[]`, `nodes_by_key`, and `edges[]`. Each session
  lane includes the Session card payload, Report/Note lists, Source/Memory hub
  counts, hub `node_keys`, optional lightweight `members[]`, and per-lane counts.
  Memory reverse links may be returned as `referencing_sessions[]`, but only from
  deterministic graph edges, edge provenance, or explicit source-session
  metadata. The endpoint never returns raw transcripts, full model output,
  request ids, secrets, or guessed reverse links from `scope_key=default` alone.
- Wiki Map uses `GET /api/v1/knowledge-graph/wiki-map` as a read-only view model
  over the active wiki snapshot. It returns `layout_mode=wiki_map`,
  `active_graph_version`, `active_snapshot`, `stale`, `stale_reason`, `stats`,
  `topics[]`, `tree_edges[]`, `cross_links[]`, and `last_refresh_job`. GET never
  triggers LLM work. Manual `POST /api/v1/knowledge-graph/wiki-map/refresh`
  starts a refresh job using the configured `roles.memory` model with thinking
  disabled; it returns a secret-free job DTO. `GET
  /api/v1/knowledge-graph/wiki-map/jobs/{job_id}` returns authoritative job
  state. `GET /api/v1/knowledge-graph/wiki-map/jobs/{job_id}/stream` replays
  durable `job_progress` / `job_terminal` events. Refresh uses staged snapshots
  and Session-level activation:
  successful Sessions are `fresh`, failed Sessions may be `carried` only after
  their previous subtree evidence is revalidated, and uncovered Sessions do not
  leave empty roots. A candidate activates only when it contains at least one
  fresh Session, has no orphan Session root, and its coverage over the same
  eligible Session set is not lower than the current active snapshot. All-failed,
  empty, orphan-root, and coverage-regressing outputs retain the previous active
  Wiki Map. Snapshot/job `stats` may add optional `eligible_sessions`,
  `fresh_sessions`, `carried_sessions`, `uncovered_sessions`,
  `candidate_coverage`, `active_coverage`, `activation_decision`,
  `activation_reason`, and recomputed quality metrics such as `topic_count`,
  `max_depth`, and `evidence_coverage`. Session-root attrs may add optional
  `generation_state`, `stale`, `stale_reason`, `source_snapshot_id`,
  `topic_count`, and `last_refreshed_at`; old snapshots remain readable when
  these fields are absent. Cross links are conservative
  evidence links from deterministic KG evidence only; no raw transcripts, raw
  model output, request ids, full source bodies, or secrets are returned.
- Knowledge Graph rebuild status may include non-sensitive `error_code` and
  `error_message` on `last_build_job`. RQ enqueue failures are marked
  `ENQUEUE_FAILED` and must not leave `queued + backend_job_id=null` jobs that
  block later rebuild attempts; 409 is reserved for a real queued/running build.
- Graph rebuild jobs expose authoritative state at
  `GET /api/v1/knowledge-graph/jobs/{job_id}` and durable progress at
  `GET /api/v1/knowledge-graph/jobs/{job_id}/stream`. Both Graph and Wiki
  streams emit `id: <seq>`, accept `after_seq` with precedence over
  `Last-Event-ID`, replay only events after the cursor, and close after an
  immutable `succeeded|partial|failed` terminal. Pre-T58 terminal jobs with no
  event history receive one synthetic terminal frame only; GET/stream never
  fabricates historical progress or writes during reads.
- `RetrievalSettings.kg_aware_retrieval_enabled` defaults to `false`, is frozen
  for each run/checkpoint lineage, and supports Multiple, Deep, and Council
  only. KG expansion discovers canonical local, memory, or external candidates;
  it is not a fourth provenance lane and never replaces live External Search.
  Results may include additive `kg_retrieval` diagnostics with
  requested/effective/attempted/applied state.
- Memory settings use `GET/PUT /api/v1/settings/memory` and persist
  `post_turn_memory_refresh_enabled` (default `true`) plus
  `workspace_memory_refresh_strategy` (`incremental|full`, default
  `incremental`) and `memory_thinking_enabled` (default `false`). The post-turn
  switch gates only automatic memory refresh after completed/result-backed runs;
  manual Refresh and the daily/weekly schedule remain available. When enabled,
  `memory_thinking_enabled` lets Memory write paths use the selected memory model
  profile's thinking budget; when disabled, Memory write paths send
  `thinking_budget=0`. The settings payload may include the latest secret-free
  `format_probe` result, including `stale` and `stale_reason` when the memory
  role/provider/model changed after the probe was recorded. Automatic post-turn
  memory refresh is skipped when the cached probe is `failed` or `stale`.
- Memory format probing uses `POST /api/v1/settings/memory/format-probe`.
  It runs a synthetic prompt against the current `roles.memory` binding, never
  reads real sessions, tries strict JSON Schema before JSON object / parser
  fallback, and returns only non-sensitive fields such as `status`,
  `provider_ref`, `capability`, `structured_strategy`, `checked_at`,
  `error_code`, `stale`, `stale_reason`, and bounded `diagnostics`. Format
  probes always keep thinking disabled to verify baseline JSON compatibility.
- Memory research profile status (`GET/POST /api/v1/memory/dream`) uses durable
  states `queued|running|cancelling|cancelled|completed|error`. Only one
  `queued|running|cancelling` Dream may exist per workspace; a conflicting start
  returns `409 MEMORY_DREAM_RUNNING` with the active Dream id. The payload may
  include `backend`, `backend_job_id`, `updated_at`, `cancel_requested_at`, and
  non-sensitive observability fields: `error_code`, `failed_stage`,
  `failed_session_id`, `attempted_sessions`, `successful_sessions`,
  `failed_sessions`, `user_profile_attempted`, `user_profile_updated`,
  `user_profile_failed`, `failure_count`, `workspace_skipped`, `provider_ref`,
  `structured_strategy`, `response_shape`, `finish_reason`, `validation_error`,
  `incomplete`, `incomplete_reasons`, `include_facts`, `accepted`, `skipped`,
  and `fact_source`. Partial session failures keep
  `status=completed` with `error_code=PARTIAL_FAILURE`; all-failed refreshes use
  `status=error` without exposing model raw output.
- `POST /api/v1/memory/dream/{dream_run_id}/cancel` requests best-effort
  cancellation for queued/running background Dreams. Terminal cancellation is
  idempotent; missing ids return `404 MEMORY_DREAM_NOT_FOUND`. Synchronous
  `background=false` Dreams are not asynchronously cancellable.
- Session-layer management uses `GET /api/v1/memories/session`. It accepts
  `workspace_key`, optional `session_id`/`query`,
  `include_archived_sessions`, `include_archived_memories`, `limit`, and
  `offset`, and returns non-empty Session groups with title/activity/archive
  metadata plus bounded memory rows. This endpoint never invokes Recall and
  never updates `last_used_at`. Session memories support archive, restore, and
  delete through existing lifecycle endpoints; PATCH, pin, TTL, and optimizer
  remain workspace/user-only.
- Memory write audit uses `GET /api/v1/memory/write-audit` and returns the latest
  secret-free write status for probe/post-turn/dream/fact paths. The payload may
  include `attempted`, `succeeded`, `skipped`, `failed`, `error_code`,
  `checked_at`, `source`, `provider_ref`, `stale`, and
  `incomplete_reasons`; it must not include raw model output, full transcripts,
  upstream request ids, or secrets.

## Error Codes

| Code | HTTP | Meaning |
|------|------|---------|
| `INVALID_REQUEST` | 400 | Invalid input or unsupported operation. |
| `UNAUTHORIZED` | 401 | Missing or invalid bearer token in token auth mode. |
| `PROVIDER_NOT_CONFIGURED` | 409 | Required LLM/search/provider configuration is absent. |
| `SEARCH_PROVIDER_STATE_CONFLICT` | 409 | Provider config revision changed; refresh before retrying Save/Toggle. |
| `CREDENTIAL_SOURCE_READ_ONLY` | 409 | An env/secure-owned credential cannot be overwritten or removed here. |
| `GROUNDING_REQUIRED` | 422 | A grounded answer could not meet citation requirements. |
| `APP_BUDGET_EXCEEDED` | 409 | Request exceeds the configured application budget. |
| `SEARCH_PROVIDER_DOWN` | 502 | Live search providers were unavailable. |
| `RUN_TIMEOUT` | 504 | Run timed out before a complete answer could be produced. |
| `RUN_CANCELLED` | 409 | Run was cancelled by the user. |
| `RUN_NOT_FOUND` | 404 | Run id is unknown. |
| `RUN_NOT_PAUSABLE` | 409 | Run is not a running Deep run at a checkpoint-safe stage. |
| `CHECKPOINT_NOT_FOUND` | 404 | The Deep run has no persisted checkpoint. |
| `CHECKPOINT_INVALID` | 422 | Checkpoint schema, hash, size, or secret-safety validation failed. |
| `CHECKPOINT_STATE_CONFLICT` | 409 | Parent/session/revision/disposition no longer permits continuation. |
| `CHECKPOINT_ALREADY_CONTINUED` | 409 | A different request already claimed the checkpoint's sole child. |
| `SESSION_NOT_FOUND` | 404 | Session id is unknown. |
| `SESSION_ARCHIVED` | 410 | Session is archived and cannot be used as active. |
| `MEMORY_PINNED` | 409 | A pinned Memory cannot receive a TTL or be archived until unpinned. |
| `KNOWLEDGE_GRAPH_BUILD_RUNNING` | 409 | A knowledge graph rebuild is already queued or running. |
| `KNOWLEDGE_GRAPH_WIKI_REFRESH_RUNNING` | 409 | A Wiki Map refresh is already queued or running. |
| `UPSTREAM_ERROR` | 502 | Upstream provider failed with a redacted message. |
| `INTERNAL_ERROR` | 500 | Unexpected internal failure. |

## Run Status And SSE

Persisted run statuses are `queued`, `running`, `pausing`, `paused`, `completed`,
`completed_degraded`, `partial_timeout`, `error`, and `cancelled`. `pausing` is a
non-terminal Deep state waiting for the next safe checkpoint; `paused` is a terminal
parent state that may create one child run. A paused parent has no RunResult, citations,
continuity messages, Memory, report, KG/Wiki projection, or semantic enrichment.
`completed_degraded` is terminal and
contains a grounded result with explicit degradation metadata. A `partial_timeout`
run is terminal and returned with HTTP 200 only when the persisted result had
already passed the normal grounding gate; an arbitrary draft or merely retrieved
sources are insufficient. Otherwise the draft is reset and timeout is stored as
`error + RUN_TIMEOUT`.

Run creation uses `POST /api/v1/runs`. The optional `search_options` object is
secret-free and may contain `include_domains[]`, `exclude_domains[]`,
`freshness` (`pd|pw|pm|py`), and `extra_subqueries[]`; empty options are treated
as omitted. Domain/freshness filters are applied to live search calls, and
`extra_subqueries` are appended after LLM-generated Multiple Search subqueries,
deduped case-insensitively with the original query retained, then capped by the run's search budget. Normalized
options are stored on `runs.metadata.search_options`.

SSE events are additive. Current public events include `workflow_start`,
`node_status`, `budget_snapshot`, `search_degraded`, `answer_delta`,
`answer_reset`, `pause_requested`, `checkpoint_saved`, `workflow_end`, and error
terminal events. `search_degraded` uses payload
`{"code":"SEARCH_PROVIDER_DOWN","fallback_tier":"indexed_external|none|cache|snippet_only"}`.
`answer_reset.reason` includes `deadline_exceeded` for an expired unverified draft.

Deep checkpoint semantics:

- `POST /runs/{id}/pause` uses Store CAS. It is accepted only for a running Deep
  run before the final-writer fence, then persists at `plan_ready`,
  `round_complete`, or `final_evidence_ready`.
- `GET /runs/{id}/checkpoint` returns secret-free integrity/readiness, stage,
  revision, disposition, lineage, reason codes, and cumulative budget summary;
  it never returns `payload_json`.
- `POST /runs/{id}/continue` accepts checkpoint revision, idempotency key,
  `continue|revise`, optional incremental budget, and a pending-only revised
  section list. It creates one child with a fresh absolute deadline while
  retaining cumulative lineage usage and the one-remediation limit.
- `POST /runs/{id}/checkpoint/abandon` retains the historical checkpoint but
  prevents continuation. Service restart never auto-continues a checkpoint.
- `GET/PATCH /settings/deep-checkpoint` manages
  `auto_checkpoint_enabled`; the default is `false`.

## Route Manifest

The table below is intentionally machine-readable for drift tests.

| Method | Path | Domain |
|--------|------|--------|
| `GET` | `/api/v1/health` | Health |
| `GET` | `/api/v1/providers` | Providers |
| `GET` | `/api/v1/knowledge-graph/status` | Knowledge Graph |
| `GET` | `/api/v1/knowledge-graph/graph` | Knowledge Graph |
| `GET` | `/api/v1/knowledge-graph/session-map` | Knowledge Graph |
| `GET` | `/api/v1/knowledge-graph/wiki-map` | Knowledge Graph |
| `POST` | `/api/v1/knowledge-graph/wiki-map/refresh` | Knowledge Graph |
| `GET` | `/api/v1/knowledge-graph/wiki-map/jobs/{job_id}` | Knowledge Graph |
| `GET` | `/api/v1/knowledge-graph/wiki-map/jobs/{job_id}/stream` | Knowledge Graph |
| `GET` | `/api/v1/knowledge-graph/nodes/{node_id}` | Knowledge Graph |
| `POST` | `/api/v1/knowledge-graph/rebuild` | Knowledge Graph |
| `GET` | `/api/v1/knowledge-graph/jobs/{job_id}` | Knowledge Graph |
| `GET` | `/api/v1/knowledge-graph/jobs/{job_id}/stream` | Knowledge Graph |
| `PUT` | `/api/v1/providers` | Providers |
| `GET` | `/api/v1/providers/llm` | Providers |
| `PUT` | `/api/v1/providers/llm` | Providers |
| `DELETE` | `/api/v1/providers/llm/{provider_id}` | Providers |
| `PATCH` | `/api/v1/providers/llm/{provider_id}` | Providers |
| `GET` | `/api/v1/providers/llm/{provider_id}/models` | Providers |
| `POST` | `/api/v1/providers/llm/{provider_id}/models` | Providers |
| `PATCH` | `/api/v1/providers/llm/{provider_id}/models/{model_id}` | Providers |
| `DELETE` | `/api/v1/providers/llm/{provider_id}/models/{model_id}` | Providers |
| `PUT` | `/api/v1/providers/roles` | Providers |
| `POST` | `/api/v1/providers/models` | Providers |
| `POST` | `/api/v1/providers/test` | Providers |
| `POST` | `/api/v1/tools/test` | Providers |
| `POST` | `/api/v1/settings/search-providers/save` | Settings |
| `PATCH` | `/api/v1/settings/search-providers/enabled` | Settings |
| `GET` | `/api/v1/sessions` | Sessions |
| `GET` | `/api/v1/sessions/archived` | Sessions |
| `POST` | `/api/v1/sessions` | Sessions |
| `PATCH` | `/api/v1/sessions/{session_id}` | Sessions |
| `POST` | `/api/v1/sessions/{session_id}/archive` | Sessions |
| `POST` | `/api/v1/sessions/{session_id}/restore` | Sessions |
| `DELETE` | `/api/v1/sessions/{session_id}` | Sessions |
| `GET` | `/api/v1/sessions/{session_id}/messages` | Sessions |
| `GET` | `/api/v1/settings/context-budget` | Settings |
| `PUT` | `/api/v1/settings/context-budget` | Settings |
| `GET` | `/api/v1/settings/retrieval` | Settings |
| `PATCH` | `/api/v1/settings/retrieval` | Settings |
| `GET` | `/api/v1/settings/deep-checkpoint` | Settings |
| `PATCH` | `/api/v1/settings/deep-checkpoint` | Settings |
| `GET` | `/api/v1/settings/search-usage` | Settings |
| `PATCH` | `/api/v1/settings/search-usage` | Settings |
| `GET` | `/api/v1/settings/post-run-notes` | Settings |
| `PUT` | `/api/v1/settings/post-run-notes` | Settings |
| `GET` | `/api/v1/settings/memory` | Settings |
| `PUT` | `/api/v1/settings/memory` | Settings |
| `POST` | `/api/v1/settings/memory/format-probe` | Settings |
| `GET` | `/api/v1/settings/knowledge-graph` | Settings |
| `PUT` | `/api/v1/settings/knowledge-graph` | Settings |
| `DELETE` | `/api/v1/cache` | Settings |
| `POST` | `/api/v1/context-budget/recommend` | Settings |
| `POST` | `/api/v1/runs/estimate-cost` | Runs |
| `POST` | `/api/v1/runs` | Runs |
| `GET` | `/api/v1/runs/{run_id}` | Runs |
| `POST` | `/api/v1/runs/{run_id}/confidence/acknowledge` | Runs |
| `POST` | `/api/v1/runs/{run_id}/regenerate` | Runs |
| `POST` | `/api/v1/runs/{run_id}/cancel` | Runs |
| `POST` | `/api/v1/runs/{run_id}/pause` | Runs |
| `GET` | `/api/v1/runs/{run_id}/checkpoint` | Runs |
| `POST` | `/api/v1/runs/{run_id}/continue` | Runs |
| `POST` | `/api/v1/runs/{run_id}/checkpoint/abandon` | Runs |
| `GET` | `/api/v1/runs/{run_id}/stream` | Runs |
| `GET` | `/api/v1/memories` | Memory |
| `GET` | `/api/v1/memories/archived` | Memory |
| `GET` | `/api/v1/memories/session` | Memory |
| `GET` | `/api/v1/memories/{memory_id}` | Memory |
| `POST` | `/api/v1/memories/cleanup` | Memory |
| `POST` | `/api/v1/memories/batch-expiry` | Memory |
| `POST` | `/api/v1/memories/optimize/preview` | Memory |
| `POST` | `/api/v1/memories/optimize/apply` | Memory |
| `GET` | `/api/v1/sessions/{session_id}/history/search` | Sessions |
| `POST` | `/api/v1/memories` | Memory |
| `POST` | `/api/v1/memories/{memory_id}/archive` | Memory |
| `POST` | `/api/v1/memories/{memory_id}/restore` | Memory |
| `PATCH` | `/api/v1/memories/{memory_id}` | Memory |
| `DELETE` | `/api/v1/memories/{memory_id}` | Memory |
| `GET` | `/api/v1/memory/dream` | Memory |
| `GET` | `/api/v1/memory/write-audit` | Memory |
| `POST` | `/api/v1/memory/dream` | Memory |
| `POST` | `/api/v1/memory/dream/{dream_run_id}/cancel` | Memory |
| `PUT` | `/api/v1/memory/dream/schedule` | Memory |
| `POST` | `/api/v1/index/url` | Index |
| `POST` | `/api/v1/index/rebuild` | Index |
| `GET` | `/api/v1/index/documents` | Index |
| `GET` | `/api/v1/library/documents` | Library |
| `POST` | `/api/v1/library/documents/import` | Library |
| `DELETE` | `/api/v1/library/documents/{document_id}` | Library |
| `GET` | `/api/v1/library/notes` | Library |
| `POST` | `/api/v1/library/notes` | Library |
| `GET` | `/api/v1/library/notes/{note_id}/provenance` | Library |
| `GET` | `/api/v1/library/reports` | Library |
| `POST` | `/api/v1/library/reports` | Library |
| `POST` | `/api/v1/library/reports/upload` | Library |
| `GET` | `/api/v1/library/reports/{report_id}` | Library |
| `GET` | `/api/v1/library/sources/{document_id}/sessions` | Library |
| `GET` | `/api/v1/library/sources/{document_id}/snapshots` | Library |
| `GET` | `/api/v1/library/snapshots/{snapshot_id}` | Library |
| `GET` | `/api/v1/notes` | Notes |
| `POST` | `/api/v1/notes` | Notes |
| `POST` | `/api/v1/tools/enhance` | Tools |

## Authoritative Sources

- `deepresearch_core/baseline/api.py`
- `deepresearch_core/baseline/errors.py`
- `deepresearch_core/baseline/service.py`
- `apps/web/src/lib/types.ts`


### T59 first-generation advisory verification (2026-11-03)

Report delivery and verification have independent lifecycles. A terminal report is immediately available with `result.verification.status=running`. Poll `GET /api/v1/runs/{run_id}` until verification is `completed` or `failed`; report SSE still ends at `workflow_end`. Verification may update only confidence metadata and conflicts, never withdraw or replace the delivered text/citations. `result.answer.confidence_label` is unset while verification runs. Low confidence uses `result.confidence_warning={reasons,acknowledged}`. An interrupted verifier becomes `failed` with low confidence; export/save remain available.

`POST /api/v1/runs/{run_id}/confidence/acknowledge` idempotently accepts the warning for this report and returns `{acknowledged,acknowledged_at}`. It does not change confidence. `POST /api/v1/runs/{run_id}/regenerate` creates a new run in the same session with the original request options, previous canonical evidence and a fresh external search; it returns `{run_id,session_id,status}` and does not copy the parent's acknowledgement. New run metadata contains a secret-free request snapshot and server-owned regeneration context.
