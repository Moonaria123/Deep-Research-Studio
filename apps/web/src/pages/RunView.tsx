import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { AlertCircle, Ban, CheckCircle2, CirclePause, LoaderCircle, Play, RotateCcw, XCircle } from "lucide-react";
import type {
  CouncilMember,
  DeepCheckpointSummary,
  EvidenceSource,
  GroundedAnswer,
  RunResult,
  TemporalCoverage,
  WriterDiagnostics,
} from "@/lib/types";
import { api, ApiError } from "@/lib/api";
import { citationIdsToSourceIndexes, sameIdSet } from "@/lib/claimMap";
import { downloadTextFile, runResultToMarkdown, slugifyQuery } from "@/lib/exportMarkdown";
import { evidenceKey } from "@/lib/evidenceRoutes";
import { useRun } from "@/hooks/useRun";
import { useGsapEntrance } from "@/hooks/useGsapEntrance";
import { RunHeader } from "@/components/RunHeader";
import { RunBudgetStrip } from "@/components/RunBudgetStrip";
import { DeepEvidenceMatrix } from "@/components/DeepEvidenceMatrix";
import { SourcesRail } from "@/components/SourcesRail";
import { AnswerBody } from "@/components/AnswerBody";
import { ResearchProcessPanel } from "@/components/ResearchProcessPanel";
import { MultipleLanesPanel } from "@/components/MultipleLanesPanel";
import { CouncilMatrixTables } from "@/components/CouncilMatrixTables";
import { CouncilReportHeader } from "@/components/CouncilReportHeader";
import { parseCouncilSections } from "@/lib/parseCouncilSections";
import { ProgressTimeline } from "@/components/ProgressTimeline";
import { ConfidenceMeter } from "@/components/ConfidenceMeter";
import { ConfidenceTag } from "@/components/ConfidenceTag";
import { LimitationsPanel } from "@/components/LimitationsPanel";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { IS_DEMO } from "@/lib/runtime";
import { Spinner } from "@/components/ui/spinner";
import { Dialog } from "@/components/ui/dialog";

const T10_ERROR_COPY: Record<string, string> = {
  RUN_TIMEOUT: "The run timed out before it could complete. Try a narrower question or retry.",
  SEARCH_PROVIDER_DOWN: "Live search providers are unavailable. The run may need provider settings or a retry later.",
  APP_BUDGET_EXCEEDED: "The request exceeded the configured app budget. Adjust the budget or narrow the request.",
  GROUNDING_REQUIRED:
    "The run could not establish enough relevant, citable support. Clarify the topic or source requirement and try again.",
};

const UNVERIFIED_REASON_COPY: Record<string, string> = {
  evidence_insufficient: "Available sources did not establish this item.",
  verification_incomplete: "Verification did not complete for this item.",
  source_conflict: "The available sources remain in conflict.",
  unsupported_claim: "The adopted evidence did not support this claim.",
};

const CONFIDENCE_WARNING_COPY: Record<string, string> = {
  evidence_insufficient: "部分内容尚缺乏充分证据支持。",
  verification_incomplete: "部分内容未能完成核验。",
  verification_interrupted: "核验流程未完整结束。",
  source_conflict: "不同来源之间仍存在冲突。",
  unsupported_claim: "部分结论未获得现有证据支持。",
  coverage_below_threshold: "部分结论的证据覆盖不足。",
  zero_citable_evidence: "当前报告缺少可核查的引用来源。",
  invalid_citation_binding: "部分结论与引用来源的对应关系需要人工核查。",
  council_member_fallback: "模型委员会未形成完整共识，报告采用了可用成员结果。",
  output_shape_invalid: "报告结构未完全符合预期，关键内容需要核查。",
  unknown_only: "现有证据尚不足以确认部分结论。",
};

const UNKNOWN_CONFIDENCE_WARNING = "部分内容尚缺乏充分证据支持，请核查关键结论。";

function ConfidenceVerificationStatus({
  result,
  acknowledged,
  onWarningClick,
}: {
  result: RunResult;
  acknowledged: boolean;
  onWarningClick: () => void;
}) {
  const verificationStatus = result.verification?.status;
  if (verificationStatus === "pending" || verificationStatus === "running") {
    return (
      <div className="mt-3 flex items-center gap-2 border-t border-border pt-3 text-sm text-text-muted" role="status" aria-label="正在核验报告">
        <LoaderCircle className="h-4 w-4 animate-spin text-accent" aria-hidden="true" />
        <span>正在核验报告置信度…</span>
      </div>
    );
  }

  const lowConfidence =
    verificationStatus === "failed" ||
    Boolean(result.confidence_warning?.reasons.length) ||
    result.answer.confidence_label === "uncertain";
  const label = verificationStatus === "failed" ? "uncertain" : result.answer.confidence_label;

  return (
    <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-border pt-3" aria-label="报告置信度">
      <span className="text-xs text-text-faint">置信度</span>
      {label ? <ConfidenceTag label={label} /> : result.confidence != null ? (
        <Badge>{Math.round(result.confidence * 100)}%</Badge>
      ) : (
        <Badge>待评估</Badge>
      )}
      {lowConfidence ? (
        <button
          type="button"
          onClick={onWarningClick}
          aria-label={acknowledged ? "低置信度风险已确认" : "查看低置信度警告"}
          title={acknowledged ? "已确认并接受低置信度风险" : "查看低置信度警告"}
          className={acknowledged
            ? "inline-flex h-6 w-6 items-center justify-center rounded-full bg-surface-muted text-text-faint focus-ring"
            : "inline-flex h-6 w-6 items-center justify-center rounded-full bg-danger text-white focus-ring"}
        >
          <AlertCircle className="h-4 w-4" aria-hidden="true" />
        </button>
      ) : null}
    </div>
  );
}

function collectSources(result?: RunResult): EvidenceSource[] {
  if (!result) return [];
  if (result.sources !== undefined) return result.sources;
  if (result.lanes?.length) {
    const seen = new Map<string, EvidenceSource>();
    result.lanes.forEach((lane) => lane.sources.forEach((s) => seen.set(evidenceKey(s), s)));
    return [...seen.values()];
  }
  return [];
}

function WriterDiagnosticsPanel({ diagnostics }: { diagnostics: WriterDiagnostics }) {
  const applied = Object.keys(diagnostics.call_options?.applied ?? {});
  const ignored = Object.keys(diagnostics.call_options?.ignored ?? {});
  const visibleTarget = diagnostics.call_options?.visible_output_target;
  const reasoningReserve = diagnostics.call_options?.reasoning_reserve_tokens;
  const providerCap = diagnostics.call_options?.provider_output_cap;
  return (
    <Card data-anim aria-label="Writer diagnostics">
      <CardContent className="pt-5">
        <details>
          <summary className="cursor-pointer text-sm font-semibold text-text-body focus-ring">
            Writer diagnostics · {diagnostics.classification}
          </summary>
          <div className="mt-3 grid gap-2 text-xs text-text-muted sm:grid-cols-2">
            <p>
              Provider: {diagnostics.provider || "unknown"}
              {diagnostics.model ? ` · ${diagnostics.model}` : ""}
            </p>
            <p>
              Finish: {diagnostics.finish_reason || "unavailable"} · visible {diagnostics.visible_chars ?? 0} chars
            </p>
            <p>Fallback: {diagnostics.fallback_scope || "none"}</p>
            <p>Attempts: {diagnostics.attempts?.length ?? 0}</p>
            {providerCap != null ? (
              <p className="sm:col-span-2">
                Output cap: visible target {visibleTarget ?? "unknown"} + reasoning reserve{" "}
                {reasoningReserve ?? "unknown"} = provider cap {providerCap} tokens
              </p>
            ) : null}
            <p className="sm:col-span-2">
              Parameters applied: {applied.join(", ") || "none"} · ignored: {ignored.join(", ") || "none"}
            </p>
          </div>
        </details>
      </CardContent>
    </Card>
  );
}

function TemporalScopeNotice({ coverage }: { coverage?: TemporalCoverage }) {
  if (!coverage?.statement?.trim()) return null;
  return (
    <div className="rounded-lg border border-border bg-bg-subtle p-4" role="note" aria-label="Temporal scope">
      <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-text-muted">正文时效范围声明</p>
      <p className="text-sm leading-6 text-text-body">{coverage.statement}</p>
    </div>
  );
}

function RunDisclosures({
  answer,
  temporalCoverage,
}: {
  answer: GroundedAnswer;
  temporalCoverage?: TemporalCoverage;
}) {
  return (
    <>
      {answer.conflicts?.length ? (
        <aside className="rounded-lg border border-warning/50 bg-warning/10 p-4" aria-label="证据冲突附注">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">证据冲突附注</p>
          <ul className="space-y-2 text-sm leading-6 text-text-body">
            {answer.conflicts.map((conflict) => (
              <li key={conflict.id}>
                <span className="font-medium">{conflict.claim}</span>：{conflict.reason}
              </li>
            ))}
          </ul>
        </aside>
      ) : null}
      {temporalCoverage?.note_required ? (
        <aside className="rounded-lg border border-warning/50 bg-warning/10 p-4" aria-label="时效性附注">
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-text-muted">时效性附注</p>
          <p className="text-sm leading-6 text-text-body">{temporalCoverage.statement}</p>
          {temporalCoverage.reason_codes.length ? (
            <p className="mt-1 text-xs text-text-faint">{temporalCoverage.reason_codes.join(" · ")}</p>
          ) : null}
        </aside>
      ) : null}
    </>
  );
}

export function Answer({
  answer,
  variant = "report",
  activeIds,
  onClaimFocus,
  onClaimPin,
  temporalCoverage,
  compact = false,
}: {
  answer: GroundedAnswer;
  variant?: "report" | "council";
  activeIds?: string[];
  onClaimFocus?: (ids: string[] | null) => void;
  onClaimPin?: (ids: string[]) => void;
  temporalCoverage?: TemporalCoverage;
  compact?: boolean;
}) {
  // Council consensus is already structured around Agreement/Disagreement/Synthesis,
  // so suppress the derived Executive summary / Key takeaways blocks to avoid a false
  // headline lifted from an Agreement bullet.
  const showStructure = variant !== "council" && !compact;
  const takeaways = showStructure ? answer.key_takeaways ?? [] : [];
  return (
    <Card data-anim>
      <CardContent className="space-y-4 pt-5">
        {answer.confidence_label && (
          <div className="flex items-center justify-between gap-3">
            <ConfidenceTag label={answer.confidence_label} />
          </div>
        )}
        <TemporalScopeNotice coverage={temporalCoverage} />
        {showStructure && answer.summary && (
          <div className="rounded-lg border border-border bg-bg-subtle p-4">
            <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-text-muted">Executive summary</p>
            <p className="text-[15px] leading-7 text-text-body">{answer.summary}</p>
          </div>
        )}
        {takeaways.length > 0 && (
          <div className="rounded-lg border border-border p-4">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">Key takeaways</p>
            <ul className="ml-5 list-disc space-y-1 text-[15px] leading-7 text-text-body">
              {takeaways.map((t, i) => (
                <li key={i}>
                  <AnswerBody text={t} citations={answer.citations} />
                </li>
              ))}
            </ul>
          </div>
        )}
        <AnswerBody
          text={answer.text}
          citations={answer.citations}
          claimMap={answer.claim_map}
          activeIds={activeIds}
          onClaimFocus={onClaimFocus}
          onClaimPin={onClaimPin}
        />
        <RunDisclosures answer={answer} temporalCoverage={temporalCoverage} />
        <LimitationsPanel text={answer.limitations} citations={answer.citations} />
        <p className="text-xs text-text-faint">
          {answer.citations.length} sources cited · {Math.round((answer.coverage ?? 0) * 100)}% citation coverage
        </p>
      </CardContent>
    </Card>
  );
}

export function CouncilView({
  consensus,
  members,
  confidence,
  breakdown,
  strength,
  temporalCoverage,
}: {
  consensus?: GroundedAnswer;
  members: CouncilMember[];
  confidence?: number;
  breakdown?: Record<string, number>;
  strength?: string;
  temporalCoverage?: TemporalCoverage;
}) {
  const sections = consensus ? parseCouncilSections(consensus) : null;
  return (
    <div className="space-y-4">
      {consensus && consensus.text.trim() && (
        <div className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-text-muted">Council consensus</p>
          <CouncilReportHeader members={members} strength={strength} />
          {sections ? (
            <Card data-anim>
              <CardContent className="space-y-4 pt-5">
                {consensus.confidence_label && <ConfidenceTag label={consensus.confidence_label} />}
                <TemporalScopeNotice coverage={temporalCoverage} />
                <CouncilMatrixTables sections={sections} citations={consensus.citations} />
                <RunDisclosures answer={consensus} temporalCoverage={temporalCoverage} />
                <LimitationsPanel text={consensus.limitations} citations={consensus.citations} />
                <p className="text-xs text-text-faint">
                  {consensus.citations.length} sources cited · {Math.round((consensus.coverage ?? 0) * 100)}% citation coverage
                </p>
              </CardContent>
            </Card>
          ) : (
            <Answer answer={consensus} variant="council" temporalCoverage={temporalCoverage} />
          )}
        </div>
      )}
      {confidence != null && <ConfidenceMeter value={confidence} breakdown={breakdown} strength={strength} />}
      <div className="space-y-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-text-muted">Member perspectives</p>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {members.map((m, i) => (
            <Card key={`${m.model}-${i}`} data-anim>
              <CardHeader>
                <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
                  <Badge variant="accent">{m.model}</Badge>
                  {m.perspective && <Badge variant="default">{m.perspective}</Badge>}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <AnswerBody text={m.answer.text} citations={m.answer.citations} />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}

function checkpointIdempotencyKey(runId: string, action: "continue" | "revise"): string {
  const random = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
  return `checkpoint-${runId}-${action}-${random}`;
}

function DeepCheckpointPanel({
  runId,
  onContinued,
}: {
  runId: string;
  onContinued: (childRunId: string) => void;
}) {
  const [checkpoint, setCheckpoint] = useState<DeepCheckpointSummary>();
  const [remainingPlan, setRemainingPlan] = useState("");
  const [additionalSearches, setAdditionalSearches] = useState(0);
  const [busy, setBusy] = useState<"continue" | "revise" | "abandon">();
  const [error, setError] = useState<string>();
  const [abandonArmed, setAbandonArmed] = useState(false);
  const idempotencyKeys = useRef<Partial<Record<"continue" | "revise", string>>>({});

  useEffect(() => {
    let disposed = false;
    api
      .getRunCheckpoint(runId)
      .then((next) => {
        if (!disposed) setCheckpoint(next);
      })
      .catch((caught) => {
        if (!disposed) setError(caught instanceof ApiError ? caught.message : "Checkpoint unavailable.");
      });
    return () => {
      disposed = true;
    };
  }, [runId]);

  const resume = async (action: "continue" | "revise") => {
    if (!checkpoint || busy) return;
    const sections = remainingPlan
      .split(/\r?\n/)
      .map((item) => item.trim())
      .filter(Boolean);
    if (action === "revise" && sections.length === 0) {
      setError("Enter at least one remaining section before revising.");
      return;
    }
    setBusy(action);
    setError(undefined);
    const idempotencyKey =
      idempotencyKeys.current[action] ??
      checkpointIdempotencyKey(runId, action);
    idempotencyKeys.current[action] = idempotencyKey;
    try {
      const child = await api.continueRun(runId, {
        expected_checkpoint_revision: checkpoint.revision,
        idempotency_key: idempotencyKey,
        action,
        incremental_budget:
          additionalSearches > 0 ? { max_searches: additionalSearches } : undefined,
        remaining_plan: action === "revise" ? { sections } : undefined,
      });
      onContinued(child.run_id);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Could not continue this checkpoint.");
      setBusy(undefined);
    }
  };

  const abandon = async () => {
    if (!checkpoint || busy) return;
    if (!abandonArmed) {
      setAbandonArmed(true);
      return;
    }
    setBusy("abandon");
    setError(undefined);
    try {
      const next = await api.abandonRunCheckpoint(runId, checkpoint.revision);
      setCheckpoint(next);
      setAbandonArmed(false);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Could not abandon this checkpoint.");
    } finally {
      setBusy(undefined);
    }
  };

  if (!checkpoint) {
    return (
      <Card data-anim role="status" aria-label="Deep checkpoint">
        <CardContent className="flex items-center gap-2 pt-5 text-sm text-text-muted">
          <Spinner className="h-4 w-4" /> Loading checkpoint…
        </CardContent>
      </Card>
    );
  }

  const resumable =
    checkpoint.disposition === "ready" &&
    (checkpoint.resume_readiness === "ready" ||
      checkpoint.resume_readiness === "degraded");
  const budget = checkpoint.budget_usage;

  return (
    <Card data-anim role="status" aria-label="Deep checkpoint">
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
          <CirclePause className="h-4 w-4" /> Deep Research paused
          <Badge variant={checkpoint.resume_readiness === "degraded" ? "warning" : "accent"}>
            {checkpoint.resume_readiness}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-text-muted">
          Safe boundary <span className="font-medium text-text-body">{checkpoint.stage}</span>
          {checkpoint.updated_at
            ? ` · saved ${new Date(checkpoint.updated_at * 1000).toLocaleString()}`
            : ""}
        </p>
        {budget ? (
          <p className="text-xs text-text-faint">
            Searches {budget.searches_used ?? 0}/{budget.searches_max ?? "?"} · pages{" "}
            {budget.pages_fetched ?? 0} · recovery {budget.recovery_searches_used ?? 0}/
            {budget.recovery_searches_max ?? 1}
          </p>
        ) : null}
        {checkpoint.reason_codes.length > 0 ? (
          <p className="text-xs text-warning">
            Resume notes: {checkpoint.reason_codes.join(", ")}
          </p>
        ) : null}
        {resumable ? (
          <>
            <label className="block space-y-1 text-sm text-text-muted">
              <span>Additional search rounds (optional)</span>
              <input
                className="h-9 w-32 rounded-md border border-border bg-surface px-3 text-text-body focus-ring"
                type="number"
                min={0}
                max={32}
                value={additionalSearches}
                onChange={(event) =>
                  setAdditionalSearches(Math.max(0, Number(event.target.value) || 0))
                }
              />
            </label>
            <label className="block space-y-1 text-sm text-text-muted">
              <span>Revised remaining sections (one per line)</span>
              <textarea
                className="min-h-24 w-full resize-y rounded-md border border-border bg-surface px-3 py-2 text-text-body focus-ring"
                value={remainingPlan}
                onChange={(event) => setRemainingPlan(event.target.value)}
                placeholder="Leave unchanged to Continue, or enter a revised remaining plan."
              />
            </label>
            <div className="flex flex-wrap gap-2">
              <Button
                size="sm"
                onClick={() => void resume("continue")}
                disabled={Boolean(busy)}
              >
                <Play className="h-4 w-4" /> {busy === "continue" ? "Continuing…" : "Continue"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => void resume("revise")}
                disabled={Boolean(busy)}
              >
                <RotateCcw className="h-4 w-4" /> {busy === "revise" ? "Revising…" : "Revise remaining plan"}
              </Button>
              <Button
                variant={abandonArmed ? "danger" : "outline"}
                size="sm"
                onClick={() => void abandon()}
                disabled={Boolean(busy)}
              >
                <XCircle className="h-4 w-4" />
                {busy === "abandon"
                  ? "Abandoning…"
                  : abandonArmed
                    ? "Confirm abandon"
                    : "Abandon"}
              </Button>
            </div>
          </>
        ) : (
          <p className="text-sm text-text-muted">
            This checkpoint is {checkpoint.disposition} and cannot create another child run.
          </p>
        )}
        {error ? <p className="text-sm text-danger" role="alert">{error}</p> : null}
      </CardContent>
    </Card>
  );
}

export function RunPanel({
  runId,
  embedded = false,
  onFollowUp,
  onChildRun,
}: {
  runId?: string;
  embedded?: boolean;
  onFollowUp?: (sessionId?: string) => void;
  onChildRun?: (runId: string) => void;
}) {
  const navigate = useNavigate();
  const { run, events, status, error, errorCode, draft, cancel, pause } = useRun(runId);
  const archived = errorCode === "SESSION_ARCHIVED";
  const ref = useGsapEntrance<HTMLDivElement>([status, run?.id]);
  const [saved, setSaved] = useState(false);
  const [saving, setSaving] = useState(false);
  const [copied, setCopied] = useState(false);
  const [verificationSnapshot, setVerificationSnapshot] = useState<RunResult | undefined>();
  const [confidenceDialogOpen, setConfidenceDialogOpen] = useState(false);
  const [confidenceAcknowledged, setConfidenceAcknowledged] = useState(false);
  const [confidenceAction, setConfidenceAction] = useState<"acknowledge" | "regenerate" | undefined>();
  const [confidenceActionError, setConfidenceActionError] = useState<string>();

  const result = verificationSnapshot ?? run?.result;
  const verificationStatus = result?.verification?.status;
  const sources = collectSources(result);
  const evidenceMatrix = result?.research_state?.evidence_matrix ?? result?.evidence_matrix ?? [];
  const streaming =
    status === "streaming" || status === "pausing" || status === "cancelling";
  const terminalResultStatus = run?.status ?? result?.status;
  const hasFinalAnswer = Boolean(result?.answer.text.trim());
  const canExport = Boolean(
    hasFinalAnswer &&
      terminalResultStatus &&
      ["succeeded", "completed", "completed_degraded", "partial_timeout"].includes(terminalResultStatus),
  );
  const canSave = !IS_DEMO && canExport;
  const partialTimeout = run?.status === "partial_timeout";
  const completedDegraded = run?.status === "completed_degraded";
  const effectiveLocale = run?.metadata?.effective_locale;
  const resolvedSearchUsage = run?.metadata?.resolved_search_usage ?? result?.resolved_search_usage;
  const writerDiagnosticsCandidate = run?.metadata?.writer_diagnostics ?? result?.writer_diagnostics;
  const writerDiagnostics = writerDiagnosticsCandidate?.classification
    ? writerDiagnosticsCandidate
    : undefined;
  const kgRetrieval = result?.kg_retrieval ?? run?.metadata?.kg_retrieval;
  const attemptedProviders = Array.from(
    new Set(
      (result?.retrieval_outcomes ?? [])
        .filter((item) => item.lane === "external")
        .flatMap((item) => item.providers_attempted?.length ? item.providers_attempted : [item.search_provider ?? item.provider])
        .filter(Boolean),
    ),
  );
  const skippedProviders = Array.from(
    new Set(
      (result?.retrieval_outcomes ?? [])
        .filter((item) => item.lane === "external")
        .flatMap((item) => item.providers_skipped ?? []),
    ),
  );
  const usedSearchProviders = Array.from(
    new Set([
      ...sources.map((source) => source.search_provider),
      ...(result?.retrieval_outcomes ?? []).map((item) => item.search_provider),
    ].filter(Boolean)),
  );
  const usedFetchProviders = Array.from(
    new Set([
      ...sources.map((source) => source.fetch_provider),
      ...(result?.retrieval_outcomes ?? []).flatMap((item) => item.fetch_providers ?? []),
    ].filter(Boolean)),
  );
  const compactOutput = Boolean(
    result?.research_intent && result.research_intent.output_shape !== "default",
  );
  const eligibilityCounts = (result?.evidence_eligibility ?? []).reduce(
    (counts, item) => ({ ...counts, [item.decision]: counts[item.decision] + 1 }),
    { eligible: 0, rejected: 0, unknown: 0 },
  );
  const errorMessage = archived
    ? "该会话已归档，内容已锁定。请先在已归档会话中恢复后再查看。"
    : errorCode && T10_ERROR_COPY[errorCode]
      ? T10_ERROR_COPY[errorCode]
      : (error ?? "The run failed.");

  useEffect(() => {
    setVerificationSnapshot(undefined);
    setConfidenceAcknowledged(false);
    setConfidenceDialogOpen(false);
    setConfidenceActionError(undefined);
  }, [runId]);

  useEffect(() => {
    if (result?.confidence_warning?.acknowledged) setConfidenceAcknowledged(true);
  }, [result?.confidence_warning?.acknowledged]);

  useEffect(() => {
    if (!runId || (verificationStatus !== "pending" && verificationStatus !== "running")) return;
    let disposed = false;
    let timer: number | undefined;

    const poll = async () => {
      try {
        const snapshot = await api.getRun(runId);
        if (disposed) return;
        if (snapshot.result) setVerificationSnapshot(snapshot.result);
        const nextStatus = snapshot.result?.verification?.status;
        if (nextStatus === "pending" || nextStatus === "running") {
          timer = window.setTimeout(() => void poll(), 1500);
        }
      } catch {
        if (!disposed) timer = window.setTimeout(() => void poll(), 2500);
      }
    };

    timer = window.setTimeout(() => void poll(), 1000);
    return () => {
      disposed = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [runId, verificationStatus]);

  // Claim <-> source bidirectional highlight. Hover takes precedence over the
  // pinned (clicked) selection so a quick preview never loses the pin.
  const [hoverIds, setHoverIds] = useState<string[] | null>(null);
  const [pinnedIds, setPinnedIds] = useState<string[]>([]);
  const activeIds = hoverIds ?? pinnedIds;
  const citations = result?.answer.citations ?? [];
  const highlightIndexes = citationIdsToSourceIndexes(activeIds, citations);
  const pinIds = (ids: string[]) => setPinnedIds((cur) => (sameIdSet(cur, ids) ? [] : ids));
  const sourceFocus = (index: number | null) => setHoverIds(index == null ? null : [`c${index}`]);
  const sourcePin = (index: number) => pinIds([`c${index}`]);

  if (!runId) {
    return (
      <Card className={embedded ? "min-h-[260px]" : "mx-auto mt-8 max-w-[720px]"}>
        <CardContent className="py-10 text-center text-sm text-text-muted">
          Select an assistant answer or submit a follow-up to open a run here.
        </CardContent>
      </Card>
    );
  }

  const handleExport = () => {
    if (!run || !result || !canExport) return;
    const markdown = runResultToMarkdown(run.query, result, sources);
    downloadTextFile(`${slugifyQuery(run.query)}.md`, markdown);
  };

  const handleCopy = async () => {
    if (!run || !result || !canExport) return;
    try {
      await navigator.clipboard.writeText(runResultToMarkdown(run.query, result, sources));
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  const handleSaveToMemory = async () => {
    if (!run || !result || saving || saved) return;
    setSaving(true);
    try {
      const content = `Q: ${run.query}\n\nA: ${result.answer.text}`.slice(0, 4000);
      await api.addMemory({ layer: "workspace", scope_key: "default", content, salience: 1.5 });
      setSaved(true);
    } catch {
      // Keep the button actionable so the user can retry.
    } finally {
      setSaving(false);
    }
  };

  const handleAcknowledgeConfidence = async () => {
    if (!runId || confidenceAction) return;
    setConfidenceAction("acknowledge");
    setConfidenceActionError(undefined);
    try {
      await api.acknowledgeRunConfidence(runId);
      setConfidenceAcknowledged(true);
      setConfidenceDialogOpen(false);
    } catch (actionError) {
      setConfidenceActionError(actionError instanceof ApiError ? actionError.message : "确认失败，请重试。");
    } finally {
      setConfidenceAction(undefined);
    }
  };

  const handleRegenerate = async () => {
    if (!runId || confidenceAction) return;
    setConfidenceAction("regenerate");
    setConfidenceActionError(undefined);
    try {
      const regenerated = await api.regenerateRun(runId);
      setConfidenceDialogOpen(false);
      navigate(`/run/${regenerated.run_id}`);
    } catch (actionError) {
      setConfidenceActionError(actionError instanceof ApiError ? actionError.message : "重新出具失败，请重试。");
    } finally {
      setConfidenceAction(undefined);
    }
  };

  return (
    <div ref={ref} className={embedded ? "w-full min-w-0" : "mx-auto w-full max-w-[1080px] px-6 py-8"}>
      <RunHeader
        mode={run?.mode ?? "fast"}
        title={run?.query ?? "Researching…"}
        meta={run ? `${status}${result?.confidence != null && verificationStatus !== "pending" && verificationStatus !== "running" ? ` · confidence ${Math.round(result.confidence * 100)}%` : ""}` : undefined}
        sessionId={run?.session_id}
        canExport={canExport}
        canSave={canSave}
        copied={copied}
        saved={saved}
        saving={saving}
        onExport={handleExport}
        onCopy={handleCopy}
        onSaveToMemory={handleSaveToMemory}
        onFollowUp={() => onFollowUp?.(run?.session_id)}
      />

      {verificationStatus === "completed" ? (
        <div className="mt-4 flex items-start gap-2 rounded-md border border-border bg-bg-subtle px-3 py-2 text-sm text-text-body" role="status" aria-label="核验完成">
          <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-success" aria-hidden="true" />
          <span>报告核验已完成，置信度信息已更新。</span>
        </div>
      ) : verificationStatus === "failed" ? (
        <div className="mt-4 flex items-start gap-2 rounded-md border border-warning/50 bg-warning/10 px-3 py-2 text-sm text-text-body" role="status" aria-label="核验未完成">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-warning" aria-hidden="true" />
          <span>报告已出具，但核验未能完成。请核查来源或重新出具。</span>
        </div>
      ) : null}

      <RunBudgetStrip
        events={events}
        mode={run?.mode ?? "fast"}
        budget={run?.budget}
        gapRounds={result?.gap_rounds}
        recoverySearchesMax={result?.recovery_searches_max}
        recoverySearchesUsed={result?.recovery_searches_used}
      />

      {effectiveLocale && Object.keys(effectiveLocale).length > 0 ? (
        <p className="mt-2 text-xs text-text-faint" aria-label="Search locale">
          Search locale: {[effectiveLocale.country, effectiveLocale.search_lang, effectiveLocale.ui_language]
            .filter(Boolean)
            .join(" · ")}
        </p>
      ) : null}

      {resolvedSearchUsage ? (
        <div className="mt-2 rounded-md border border-border bg-bg-subtle px-3 py-2 text-xs text-text-muted" aria-label="Search usage">
          <p>
            Search usage: <span className="font-medium text-text-body">{resolvedSearchUsage.effective_policy.usage_mode}</span>
            {` · ${resolvedSearchUsage.source} · search ${[
              ...resolvedSearchUsage.search_chain,
              ...(resolvedSearchUsage.search_fallback_tail ?? ["cache", "indexed_external"]),
            ].join(" → ") || "none"}`}
            {` · extract ${resolvedSearchUsage.extract_chain.join(" → ") || "none"}`}
          </p>
          <p className="mt-1 text-text-faint">
            Attempted {attemptedProviders.join(", ") || "none"} · selected search {usedSearchProviders.join(", ") || "none"} · selected fetch {usedFetchProviders.join(", ") || "none"}
            {[...resolvedSearchUsage.reason_codes, ...skippedProviders].length
              ? ` · skipped ${[...resolvedSearchUsage.reason_codes, ...skippedProviders].join(", ")}`
              : ""}
          </p>
          <p className="mt-1 text-text-faint">Transport fallback order does not determine evidence authority.</p>
        </div>
      ) : null}

      {kgRetrieval?.requested ? (
        <div className="mt-2 rounded-md border border-border bg-bg-subtle px-3 py-2 text-xs text-text-muted" aria-label="Knowledge graph retrieval">
          <p>
            KG candidate discovery:{" "}
            <span className="font-medium text-text-body">
              {kgRetrieval.applied
                ? "evidence adopted"
                : kgRetrieval.attempted
                  ? "checked"
                  : kgRetrieval.effective
                    ? "ready"
                    : "unavailable"}
            </span>
            {` · ${kgRetrieval.candidate_count} candidates · ${kgRetrieval.hydrated_count} hydrated`}
          </p>
          <p className="mt-1 text-text-faint">
            KG is not a fourth evidence source. Hydrated items retain local, Memory, or external provenance.
            {kgRetrieval.reason_codes.length ? ` · ${kgRetrieval.reason_codes.join(", ")}` : ""}
          </p>
        </div>
      ) : null}

      <div className={embedded ? "mt-4 grid min-w-0 grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_220px]" : "mt-6 flex gap-8"}>
        <div className="min-w-0 flex-1 space-y-4">
          {streaming && (
            <Card data-anim>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Spinner className="h-4 w-4" />{" "}
                  {status === "cancelling"
                    ? "Cancelling…"
                    : status === "pausing"
                      ? "Pausing at the next safe checkpoint…"
                      : "Researching…"}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <ProgressTimeline events={events} done={false} />
                {draft ? (
                  <div
                    className="rounded-lg border border-border bg-bg-subtle p-4"
                    role="status"
                    aria-label="Streaming answer draft"
                  >
                    <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
                      Draft · grounding in progress
                    </p>
                    <AnswerBody text={draft} citations={[]} />
                  </div>
                ) : null}
                <div className="flex flex-wrap gap-2">
                  {run?.mode === "deep" && status === "streaming" ? (
                    <Button variant="outline" size="sm" onClick={pause}>
                      <CirclePause className="h-4 w-4" /> Pause
                    </Button>
                  ) : null}
                  <Button variant="outline" size="sm" onClick={cancel} disabled={status === "cancelling"}>
                    <Ban className="h-4 w-4" /> Cancel
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {status === "paused" && runId ? (
            <DeepCheckpointPanel
              runId={runId}
              onContinued={(childRunId) => {
                if (onChildRun) onChildRun(childRunId);
                else navigate(`/run/${childRunId}`);
              }}
            />
          ) : null}

          {status === "error" && (
            <Card data-anim className="border-danger/40">
              <CardContent className="space-y-3 pt-5">
                <p role="alert" className="text-sm text-danger">
                  {errorMessage}
                </p>
                {archived && (
                  <Link
                    to="/history/archived"
                    className="inline-flex h-8 items-center rounded-md border border-border bg-surface px-3 text-sm text-text hover:bg-bg-subtle focus-ring"
                  >
                    前往已归档会话
                  </Link>
                )}
              </CardContent>
            </Card>
          )}

          {status === "cancelled" && (
            <Card data-anim className="border-border bg-bg-subtle" role="status" aria-label="Cancelled run">
              <CardContent className="space-y-1 pt-5">
                <p className="text-sm font-semibold text-text-body">Run cancelled</p>
                <p className="text-sm text-text-muted">
                  The run stopped without keeping an ungrounded answer draft.
                </p>
              </CardContent>
            </Card>
          )}

          {!streaming && writerDiagnostics ? (
            <WriterDiagnosticsPanel diagnostics={writerDiagnostics} />
          ) : null}

          {!streaming && result && (
            <>
              {result.remediation?.attempted && (
                <Card data-anim role="status" aria-label="Evidence recovery">
                  <CardContent className="space-y-1 pt-5 text-sm text-text-muted">
                    <p className="font-semibold text-text-body">Evidence recovery {result.remediation.status}</p>
                    <p>
                      {result.remediation.action === "rewrite_only"
                        ? "The answer was rewritten against the adopted evidence without another search."
                        : "Local, memory, and external evidence were checked once more with a focused query."}
                    </p>
                    <p className="text-xs text-text-faint">
                      Eligible {eligibilityCounts.eligible} · Rejected {eligibilityCounts.rejected} · Unknown{" "}
                      {eligibilityCounts.unknown} · Recovery {result.recovery_searches_used ?? 0}/
                      {result.recovery_searches_max ?? 1}
                    </p>
                  </CardContent>
                </Card>
              )}
              {partialTimeout && (
                <Card data-anim className="border-warning/50 bg-warning/10" role="status" aria-label="Partial result">
                  <CardContent className="space-y-1 pt-5">
                    <p className="text-sm font-semibold text-text-body">Partial result</p>
                    <p className="text-sm text-text-muted">
                      The run timed out, so this page shows the evidence-backed answer generated so far.
                    </p>
                  </CardContent>
                </Card>
              )}
              {completedDegraded && (
                <Card data-anim className="border-warning/50 bg-warning/10" role="status" aria-label="Degraded result">
                  <CardContent className="space-y-3 pt-5">
                    <div className="space-y-1">
                      <p className="text-sm font-semibold text-text-body">Completed with disclosed verification boundaries</p>
                      <p className="text-sm text-text-muted">
                        The report is available for use. Review the items below before relying on affected findings.
                      </p>
                    </div>
                    {result.unverified_items?.length ? (
                      <ul className="space-y-2 text-sm text-text-muted" aria-label="Unverified items">
                        {result.unverified_items.map((item, index) => (
                          <li key={`${item.question}-${index}`}>
                            <span className="font-medium text-text-body">{item.question}</span>
                            <span className="block text-xs text-text-faint">
                              {UNVERIFIED_REASON_COPY[item.reason] ?? item.reason}
                            </span>
                          </li>
                        ))}
                      </ul>
                    ) : null}
                  </CardContent>
                </Card>
              )}
              {result.mode === "council" && result.members ? (
                <CouncilView
                  consensus={result.answer}
                  members={result.members}
                  confidence={verificationStatus === "pending" || verificationStatus === "running" ? undefined : result.confidence}
                  breakdown={result.breakdown}
                  strength={result.consensus_strength}
                  temporalCoverage={result.temporal_coverage}
                />
              ) : result.mode === "deep" && result.sections ? (
                <div className="space-y-4">
                  <Answer
                    answer={result.answer}
                    compact={compactOutput}
                    temporalCoverage={result.temporal_coverage}
                    activeIds={activeIds}
                    onClaimFocus={setHoverIds}
                    onClaimPin={pinIds}
                  />
                  <DeepEvidenceMatrix rows={evidenceMatrix} />
                  <ResearchProcessPanel sections={result.sections} />
                </div>
              ) : result.mode === "multiple" && result.lanes?.length ? (
                <div className="space-y-4">
                  <Answer
                    answer={result.answer}
                    compact={compactOutput}
                    temporalCoverage={result.temporal_coverage}
                    activeIds={activeIds}
                    onClaimFocus={setHoverIds}
                    onClaimPin={pinIds}
                  />
                  <MultipleLanesPanel lanes={result.lanes} adoptedSources={result.sources} />
                </div>
              ) : (
                <Answer
                  answer={result.answer}
                  compact={compactOutput}
                  temporalCoverage={result.temporal_coverage}
                  activeIds={activeIds}
                  onClaimFocus={setHoverIds}
                  onClaimPin={pinIds}
                />
              )}
              <ConfidenceVerificationStatus
                result={result}
                acknowledged={confidenceAcknowledged || Boolean(result.confidence_warning?.acknowledged)}
                onWarningClick={() => {
                  setConfidenceActionError(undefined);
                  setConfidenceDialogOpen(true);
                }}
              />
            </>
          )}
        </div>

        <div className={embedded ? "min-w-0" : ""}>
          <SourcesRail
            sources={sources}
            highlightIndexes={highlightIndexes}
            onSourceFocus={sourceFocus}
            onSourcePin={sourcePin}
          />
        </div>
      </div>

      <Dialog
        open={confidenceDialogOpen}
        onOpenChange={setConfidenceDialogOpen}
        title="低置信度警告"
        description="当前报告已正常出具，但部分结论的证据支持或核验结果有限。"
      >
        <div className="space-y-4">
          <p className="text-sm text-text-muted">
            建议先核查关键来源，或使用当前已有证据并重新搜索补充证据后重新出具报告。
          </p>
          {result?.confidence_warning?.reasons.length ? (
            <ul className="list-disc space-y-1 pl-5 text-xs text-text-faint" aria-label="低置信度原因">
              {result.confidence_warning.reasons.map((reason) => (
                <li key={reason}>{CONFIDENCE_WARNING_COPY[reason] ?? UNKNOWN_CONFIDENCE_WARNING}</li>
              ))}
            </ul>
          ) : null}
          {confidenceActionError ? <p className="text-sm text-danger" role="alert">{confidenceActionError}</p> : null}
          <div className="flex flex-wrap justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => void handleAcknowledgeConfidence()}
              disabled={Boolean(confidenceAction)}
            >
              {confidenceAction === "acknowledge" ? "确认中…" : "确认并忽略"}
            </Button>
            <Button
              type="button"
              onClick={() => void handleRegenerate()}
              disabled={Boolean(confidenceAction)}
            >
              {confidenceAction === "regenerate" ? "重新出具中…" : "重新出具"}
            </Button>
          </div>
        </div>
      </Dialog>
    </div>
  );
}

export function RunView() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  return (
    <RunPanel
      runId={runId}
      onFollowUp={(sessionId) => navigate(sessionId ? `/session/${sessionId}` : "/")}
    />
  );
}
