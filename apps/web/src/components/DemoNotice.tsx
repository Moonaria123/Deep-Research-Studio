import { ExternalLink, Eye } from "lucide-react";
import { IS_DEMO } from "@/lib/runtime";

const GITHUB_REPOSITORY_URL = "https://github.com/Moonaria123/Deep-Research-Studio";

export function DemoNotice() {
  if (!IS_DEMO) return null;
  return (
    <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-2 border-b border-warning/40 bg-warning/10 px-3 py-2 text-center text-xs text-text-body sm:justify-between sm:px-4" role="note" aria-label="Demo preview notice">
      <div className="flex min-w-0 flex-wrap items-center justify-center gap-x-2 gap-y-1 sm:flex-1">
        <Eye className="h-4 w-4 shrink-0 text-warning" aria-hidden />
        <strong>UI preview only</strong>
        <span>Preset data · No model calls · No keys · Nothing is saved</span>
      </div>
      <a
        href={GITHUB_REPOSITORY_URL}
        target="_blank"
        rel="noreferrer"
        aria-label="View Deep Research Studio on GitHub"
        className="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-warning/40 bg-bg px-2.5 py-1 font-medium text-text transition-colors hover:border-warning/70 hover:bg-bg-subtle focus-ring"
      >
        <ExternalLink className="h-3.5 w-3.5" aria-hidden />
        GitHub
      </a>
    </div>
  );
}
