import type { EvidenceSource, RunResult } from "./types";
import { evidenceKey } from "./evidenceRoutes";

const CONFIDENCE_COPY: Record<string, string> = {
  established: "Established",
  contested: "Contested",
  uncertain: "Uncertain",
};

const UNVERIFIED_REASON_COPY: Record<string, string> = {
  evidence_insufficient: "Evidence was insufficient",
  verification_incomplete: "Verification did not complete",
  source_conflict: "Sources remain in conflict",
  unsupported_claim: "The claim was not supported by the adopted evidence",
};

function pushVerificationBoundary(lines: string[], result: RunResult): void {
  if (result.status === "partial_timeout") {
    lines.push("> **Partial result** — the run timed out before the report was complete.");
    lines.push("");
  }

  if (result.verification?.status === "pending" || result.verification?.status === "running") {
    lines.push("> **Verification pending** — confidence checks were still running when this file was exported.");
    lines.push("");
  } else if (result.verification?.status === "failed") {
    lines.push("> **Verification incomplete** — confidence checks could not finish. Review the cited sources before relying on this report.");
    lines.push("");
  } else if (result.confidence_warning?.reasons.length) {
    const acknowledgement = result.confidence_warning.acknowledged ? " Risk accepted by the user." : "";
    lines.push(`> **Low-confidence warning** — verify important findings or regenerate the report.${acknowledgement}`);
    lines.push("");
  }

  if (!result.unverified_items?.length) return;

  lines.push("## Items to verify");
  lines.push("");
  result.unverified_items.forEach((item) => {
    const reason = UNVERIFIED_REASON_COPY[item.reason] ?? item.reason;
    lines.push(`- ${item.question} — ${reason}.`);
  });
  lines.push("");
}

/** Append a dedicated ``## Limitations`` block when an answer carries one. */
function pushLimitations(lines: string[], limitations?: string): void {
  const text = limitations?.trim();
  if (!text) return;
  lines.push("## Limitations");
  lines.push("");
  lines.push(text);
  lines.push("");
}

function pushTemporalScope(lines: string[], result: RunResult): void {
  const statement = result.temporal_coverage?.statement?.trim();
  if (!statement) return;
  lines.push(`> 时效范围声明：${statement}`);
  lines.push("");
}

function pushConflictNote(lines: string[], result: RunResult): void {
  const conflicts = result.answer.conflicts ?? result.conflicts ?? [];
  if (!conflicts.length) return;
  lines.push("## 证据冲突附注");
  lines.push("");
  conflicts.forEach((conflict) => {
    lines.push(`- **${conflict.claim}**：${conflict.reason}`);
    lines.push(`  - 范围：${conflict.claim_scope}`);
    lines.push(`  - 证据：${conflict.evidence_ids.join(", ")}`);
  });
  lines.push("");
}

function pushTemporalNote(lines: string[], result: RunResult): void {
  const coverage = result.temporal_coverage;
  if (!coverage?.note_required) return;
  lines.push("## 时效性附注");
  lines.push("");
  lines.push(coverage.statement.trim());
  if (coverage.reason_codes.length) lines.push(`原因代码：${coverage.reason_codes.join(", ")}`);
  lines.push("");
}

/**
 * Render a run result as a self-contained Markdown document: the grounded answer,
 * any deep-research sections or council members, a separate limitations block, and a
 * numbered sources list.
 */
export function runResultToMarkdown(query: string, result: RunResult, sources: EvidenceSource[]): string {
  const lines: string[] = [];
  lines.push(`# ${query || "DeepResearch report"}`);
  lines.push("");
  lines.push(`> Mode: ${result.mode} · Status: ${result.status}`);
  if (result.confidence != null) {
    lines.push(`> Confidence: ${Math.round(result.confidence * 100)}%`);
  }
  const label = result.answer.confidence_label;
  if (label && CONFIDENCE_COPY[label]) {
    lines.push(`> Confidence rating: ${CONFIDENCE_COPY[label]}`);
  }
  lines.push("");
  pushVerificationBoundary(lines, result);
  pushTemporalScope(lines, result);

  if (result.members?.length) {
    if (result.answer.text.trim()) {
      lines.push("## Consensus");
      lines.push("");
      lines.push(result.answer.text.trim());
      lines.push("");
      pushConflictNote(lines, result);
      pushTemporalNote(lines, result);
      pushLimitations(lines, result.answer.limitations);
    }
    result.members.forEach((member) => {
      lines.push(`## ${member.model}`);
      lines.push("");
      lines.push(member.answer.text.trim());
      lines.push("");
    });
  } else {
    lines.push(result.answer.text.trim());
    lines.push("");
    pushConflictNote(lines, result);
    pushTemporalNote(lines, result);
    pushLimitations(lines, result.answer.limitations);
    // Deep-research section drafts are demoted to a "Research process" appendix so the
    // exported document leads with the cohesive final report.
    if (result.sections?.length) {
      lines.push("## Research process");
      lines.push("");
      result.sections.forEach((section, index) => {
        lines.push(`### ${index + 1}. ${section.title}`);
        lines.push("");
        lines.push(section.answer.text.trim());
        lines.push("");
      });
    }
    // Multiple-mode parallel lanes are appended so the export documents the search
    // strategy: each sub-query plus the sources retrieved in that lane.
    if (result.lanes?.length) {
      const adoptedKeys = result.sources !== undefined ? new Set(sources.map(evidenceKey)) : null;
      lines.push("## Search lanes");
      lines.push("");
      result.lanes.forEach((lane, index) => {
        lines.push(`### ${index + 1}. ${lane.query}`);
        lines.push("");
        const laneSources = adoptedKeys
          ? lane.sources.filter((source) => adoptedKeys.has(evidenceKey(source)))
          : lane.sources;
        if (laneSources.length) {
          laneSources.forEach((source) => {
            const title = source.title || source.url || "Source";
            lines.push(source.url ? `- [${title}](${source.url})` : `- ${title}`);
          });
        } else {
          lines.push("- (no sources)");
        }
        lines.push("");
      });
    }
  }

  if (sources.length) {
    lines.push("## Sources");
    lines.push("");
    sources.forEach((source, index) => {
      const title = source.title || source.url || `Source ${index + 1}`;
      lines.push(source.url ? `${index + 1}. [${title}](${source.url})` : `${index + 1}. ${title}`);
    });
    lines.push("");
  }

  return lines.join("\n");
}

/** Trigger a client-side download of the given text as a file. */
export function downloadTextFile(filename: string, text: string): void {
  const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

/** Build a filesystem-safe filename slug from a query. */
export function slugifyQuery(query: string, fallback = "deepresearch"): string {
  const slug = query
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60);
  return slug || fallback;
}
