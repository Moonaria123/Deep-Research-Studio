import type { KnowledgeGraphWikiMap } from "@/lib/types";
import type { WikiFreshnessSummary, WikiPartialSummary } from "@/lib/wikiPartialSummary";

export type WikiOperatorHint = {
  title: string;
  detail: string;
};

export function wikiPartialHint(summary: WikiPartialSummary | null): WikiOperatorHint | null {
  if (!summary) return null;
  const dominant =
    summary.dominantErrorCount > 0
      ? ` Dominant error: ${summary.dominantErrorCode} (${summary.dominantErrorCount}).`
      : "";
  return {
    title: "Wiki snapshot is partial.",
    detail: `${summary.succeeded} of ${summary.attempted} Sessions succeeded; ${summary.failed} failed and ${summary.skipped} skipped.${dominant}`,
  };
}

export function wikiErrorCodeHint(errorCode?: string | null): WikiOperatorHint | null {
  const code = String(errorCode || "").trim().toUpperCase();
  if (code === "PROVIDER_NOT_CONFIGURED") {
    return {
      title: "Memory model is not configured",
      detail: "Assign an available model to the Memory role in Settings, run Test memory format, then select Refresh Wiki.",
    };
  }
  if (code === "WIKI_LLM_FAILED") {
    return {
      title: "Wiki model call failed",
      detail: "Check the Memory model's connection, quota, and API key. If attempts=0 still appears, check the refresh job's diagnostic summary.",
    };
  }
  if (code === "NO_WIKI_TOPICS") {
    return {
      title: "No Wiki topics are ready to activate",
      detail: "Make sure the current graph contains Sessions with assistant responses. If all Sessions failed, resolve the model call failures first.",
    };
  }
  if (code === "INVALID_EVIDENCE_REF") {
    return {
      title: "Invalid evidence reference",
      detail: "This can occur with older snapshots. Select Refresh Wiki in the current version to regenerate topics using the updated set of allowed evidence.",
    };
  }
  return null;
}

export function wikiFreshnessHint(freshness: WikiFreshnessSummary | null): WikiOperatorHint | null {
  if (!freshness) return null;
  if (freshness.graphVersionMismatch) {
    return {
      title: "Wiki snapshot is from an older graph version.",
      detail: "Refresh Wiki after the graph is current to rebuild topic evidence against the active graph.",
    };
  }
  if (freshness.stale) {
    return {
      title: "Wiki snapshot is stale.",
      detail: `Reason: ${freshness.staleReason || "graph changed"}. Refresh Wiki to rebuild topic evidence.`,
    };
  }
  return null;
}

export function wikiNoCrossLinkHint({
  wikiMap,
  visibleCrossEdgeCount,
}: {
  wikiMap?: KnowledgeGraphWikiMap;
  visibleCrossEdgeCount: number;
}): WikiOperatorHint | null {
  if (visibleCrossEdgeCount > 0) return null;
  const totalCrossLinks = wikiMap?.stats.cross_links ?? wikiMap?.cross_links.length ?? 0;
  if (totalCrossLinks === 0) {
    return {
      title: "No Wiki cross-session links in this snapshot.",
      detail: wikiMap?.stale
        ? `Snapshot is stale: ${wikiMap.stale_reason || "graph changed"}. Refresh Wiki to rebuild evidence links.`
        : "Refresh Wiki after the graph is current to discover high-confidence evidence links.",
    };
  }
  return {
    title: "Wiki cross-session links are hidden by the current filter, search, or type filters.",
    detail: "Try showing all Sessions or removing search/type filters.",
  };
}

export function wikiHiddenPeerHint(title: string): WikiOperatorHint {
  return {
    title: "Peer topic is hidden by the current view.",
    detail: `${title || "This topic"} is not visible under the current Session, search, or type filters.`,
  };
}
