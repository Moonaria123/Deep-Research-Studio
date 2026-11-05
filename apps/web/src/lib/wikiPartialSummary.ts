import type { KnowledgeGraphWikiMap } from "@/lib/types";

export type WikiPartialErrorRow = {
  sessionId: string;
  errorCode: string;
  validationError: string;
  strategy: string;
};

export type WikiPartialSummary = {
  succeeded: number;
  attempted: number;
  failed: number;
  skipped: number;
  dominantErrorCode: string;
  dominantErrorCount: number;
  errors: WikiPartialErrorRow[];
  repairText: string;
};

export type WikiFreshnessSummary = {
  status: string;
  stale: boolean;
  staleReason: string;
  lastRefreshedAt?: number | null;
  graphVersionMismatch: boolean;
  snapshotGraphVersionId: string;
  activeGraphVersionId: string;
};

function numberStat(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function stringStat(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function firstWikiPartialErrors(statsList: Array<Record<string, unknown>>): unknown[] {
  for (const stats of statsList) {
    const raw = stats.errors;
    if (Array.isArray(raw) && raw.length > 0) return raw;
  }
  return [];
}

export function wikiPartialErrorCounts(...statsList: Array<Record<string, unknown>>): Record<string, number> {
  const raw = firstWikiPartialErrors(statsList);
  return raw.reduce<Record<string, number>>((acc, item) => {
    const row = item && typeof item === "object" ? (item as Record<string, unknown>) : {};
    const errorCode = stringStat(row.error_code) || "WIKI_SESSION_FAILED";
    const validationError = stringStat(row.validation_error);
    const code = errorCode || validationError || "WIKI_SESSION_FAILED";
    acc[code] = (acc[code] ?? 0) + 1;
    return acc;
  }, {});
}

export function wikiPartialErrorRows(...statsList: Array<Record<string, unknown>>): WikiPartialErrorRow[] {
  return firstWikiPartialErrors(statsList)
    .slice(0, 3)
    .map((item) => {
      const row = item && typeof item === "object" ? (item as Record<string, unknown>) : {};
      return {
        sessionId: stringStat(row.session_id),
        errorCode: stringStat(row.error_code) || "WIKI_SESSION_FAILED",
        validationError: stringStat(row.validation_error),
        strategy: stringStat(row.structured_strategy),
      };
    });
}

export function summarizeWikiPartial(wikiMap?: KnowledgeGraphWikiMap): WikiPartialSummary | null {
  if (wikiMap?.active_snapshot?.status !== "partial") return null;
  const snapshotStats = wikiMap.active_snapshot.stats ?? {};
  const mapStats = wikiMap.stats ?? {};
  const jobStats = wikiMap.last_refresh_job?.stats ?? {};
  const countStats = Object.keys(snapshotStats).length ? snapshotStats : Object.keys(mapStats).length ? mapStats : jobStats;
  const errors = wikiPartialErrorRows(jobStats, snapshotStats, mapStats);
  const counts = wikiPartialErrorCounts(jobStats, snapshotStats, mapStats);
  const dominant = Object.entries(counts).sort((a, b) => b[1] - a[1])[0] ?? ["", 0];
  const repairParts = [
    { label: "repaired", value: numberStat(countStats.evidence_ref_repaired) },
    { label: "dropped refs", value: numberStat(countStats.evidence_ref_dropped) },
    { label: "dropped topics", value: numberStat(countStats.topic_dropped) },
  ]
    .filter((part) => part.value > 0)
    .map((part) => `${part.value} ${part.label}`);
  return {
    succeeded: numberStat(countStats.succeeded_sessions ?? wikiMap.last_refresh_job?.succeeded_sessions),
    attempted: numberStat(countStats.attempted_sessions ?? wikiMap.last_refresh_job?.attempted_sessions),
    failed: numberStat(countStats.failed_sessions ?? wikiMap.last_refresh_job?.failed_sessions),
    skipped: numberStat(countStats.skipped_sessions ?? wikiMap.last_refresh_job?.skipped_sessions),
    dominantErrorCode: String(dominant[0] || "WIKI_SESSION_FAILED"),
    dominantErrorCount: Number(dominant[1] || 0),
    errors,
    repairText: repairParts.join(" · "),
  };
}

export function summarizeWikiFreshness(wikiMap?: KnowledgeGraphWikiMap): WikiFreshnessSummary | null {
  const snapshot = wikiMap?.active_snapshot;
  if (!snapshot) return null;
  const snapshotGraphVersionId = snapshot.graph_version_id || "";
  const activeGraphVersionId = wikiMap?.active_graph_version?.id || "";
  return {
    status: snapshot.status || "unknown",
    stale: !!(wikiMap?.stale || snapshot.stale),
    staleReason: wikiMap?.stale_reason || snapshot.stale_reason || "",
    lastRefreshedAt:
      snapshot.finished_at ??
      snapshot.activated_at ??
      wikiMap?.last_refresh_job?.finished_at ??
      wikiMap?.last_refresh_job?.started_at ??
      snapshot.started_at ??
      null,
    graphVersionMismatch: !!snapshotGraphVersionId && !!activeGraphVersionId && snapshotGraphVersionId !== activeGraphVersionId,
    snapshotGraphVersionId,
    activeGraphVersionId,
  };
}
