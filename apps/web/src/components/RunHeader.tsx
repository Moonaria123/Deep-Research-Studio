import { useNavigate } from "react-router-dom";
import { ArrowLeft, Download, BookmarkPlus, Check, Copy, CornerDownRight } from "lucide-react";
import type { Mode } from "@/lib/types";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";

const MODE_LABEL: Record<Mode, string> = {
  fast: "Fast Search",
  multiple: "Multiple Search",
  deep: "Deep Research",
  council: "Models Council",
};

export function RunHeader({
  mode,
  title,
  meta,
  sessionId,
  canExport = false,
  canSave = false,
  copied = false,
  saved = false,
  saving = false,
  onExport,
  onCopy,
  onSaveToMemory,
  onFollowUp,
}: {
  mode: Mode;
  title: string;
  meta?: string;
  sessionId?: string | null;
  canExport?: boolean;
  canSave?: boolean;
  copied?: boolean;
  saved?: boolean;
  saving?: boolean;
  onExport?: () => void;
  onCopy?: () => void;
  onSaveToMemory?: () => void;
  onFollowUp?: () => void;
}) {
  const navigate = useNavigate();
  return (
    <div data-anim className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-1">
          <Button variant="ghost" size="sm" onClick={() => navigate("/")} aria-label="Back">
            <ArrowLeft className="h-4 w-4" /> Back
          </Button>
          {sessionId && (
            <Button variant="ghost" size="sm" onClick={() => navigate(`/session/${sessionId}`)}>
              <CornerDownRight className="h-4 w-4" /> 返回会话
            </Button>
          )}
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2">
          <Button variant="outline" size="sm" onClick={onCopy} disabled={!canExport}>
            {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
            {copied ? "Copied" : "Copy"}
          </Button>
          <Button variant="outline" size="sm" onClick={onExport} disabled={!canExport}>
            <Download className="h-4 w-4" /> Export
          </Button>
          <Button variant="outline" size="sm" onClick={onSaveToMemory} disabled={!canSave || saving || saved}>
            {saved ? <Check className="h-4 w-4" /> : <BookmarkPlus className="h-4 w-4" />}
            {saved ? "Saved" : saving ? "Saving…" : "Save to memory"}
          </Button>
          <Button size="sm" onClick={onFollowUp}>
            <CornerDownRight className="h-4 w-4" /> Ask follow-up
          </Button>
        </div>
      </div>
      <Badge variant="accent">{MODE_LABEL[mode]}</Badge>
      <h1 className="break-words text-2xl font-bold text-text">{title}</h1>
      {meta && <p className="text-sm text-text-muted">{meta}</p>}
    </div>
  );
}
