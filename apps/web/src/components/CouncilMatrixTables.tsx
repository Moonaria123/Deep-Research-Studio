import type { ReactNode } from "react";
import { Handshake, GitFork, Sparkles, Scale } from "lucide-react";
import type { Citation } from "@/lib/types";
import type { CouncilSections } from "@/lib/parseCouncilSections";
import { AnswerBody } from "@/components/AnswerBody";

function SectionTable({
  title,
  icon,
  rows,
  citations,
}: {
  title: string;
  icon: ReactNode;
  rows: string[];
  citations: Citation[];
}) {
  if (!rows.length) return null;
  return (
    <div className="space-y-2">
      <h3 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
        {icon}
        {title}
        <span className="text-text-faint">({rows.length})</span>
      </h3>
      <div className="overflow-x-auto rounded-md border border-border">
        <table className="w-full border-collapse text-sm">
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-b border-border align-top last:border-0">
                <td className="w-10 px-3 py-2 text-text-faint">{i + 1}</td>
                <td className="px-3 py-2">
                  <AnswerBody text={row} citations={citations} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/**
 * Render a parsed council consensus as Agreement / Disagreement / Unique-findings
 * tables plus a Synthesis block, so members' converging and conflicting points read
 * as a scannable matrix rather than one long Markdown blob.
 */
export function CouncilMatrixTables({
  sections,
  citations,
}: {
  sections: CouncilSections;
  citations: Citation[];
}) {
  return (
    <div className="space-y-4">
      <SectionTable
        title="Agreement"
        icon={<Handshake className="h-3.5 w-3.5" aria-hidden="true" />}
        rows={sections.agreement}
        citations={citations}
      />
      <SectionTable
        title="Disagreement"
        icon={<GitFork className="h-3.5 w-3.5" aria-hidden="true" />}
        rows={sections.disagreement}
        citations={citations}
      />
      <SectionTable
        title="Unique findings"
        icon={<Sparkles className="h-3.5 w-3.5" aria-hidden="true" />}
        rows={sections.uniqueFindings}
        citations={citations}
      />
      {sections.synthesis && (
        <div className="space-y-2">
          <h3 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
            <Scale className="h-3.5 w-3.5" aria-hidden="true" />
            Synthesis
          </h3>
          <div className="rounded-md border border-border bg-bg-subtle/60 p-4">
            <AnswerBody text={sections.synthesis} citations={citations} />
          </div>
        </div>
      )}
    </div>
  );
}
