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
      title: "Memory 模型未配置",
      detail: "到 Settings 的 Memory 角色绑定可用模型，运行 Test memory format 后再 Refresh Wiki。",
    };
  }
  if (code === "WIKI_LLM_FAILED") {
    return {
      title: "Wiki 模型调用失败",
      detail: "检查 Memory 模型的网络、配额和密钥状态；若仍显示 attempts=0，请查看刷新 job 的诊断摘要。",
    };
  }
  if (code === "NO_WIKI_TOPICS") {
    return {
      title: "没有生成可激活的 Wiki topics",
      detail: "确认当前图谱包含有 assistant 输出的 Session；若所有 Session 都失败，先修复上游模型调用。",
    };
  }
  if (code === "INVALID_EVIDENCE_REF") {
    return {
      title: "Evidence 引用不合法",
      detail: "这是旧快照常见错误；请在当前版本重新 Refresh Wiki，让系统用新的 evidence allowlist 生成。",
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
