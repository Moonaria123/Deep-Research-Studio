import * as React from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

interface DialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title?: React.ReactNode;
  description?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

/** Minimal accessible modal: overlay + panel, Escape / backdrop / close to dismiss. */
export function Dialog({ open, onOpenChange, title, description, children, className }: DialogProps) {
  const titleId = React.useId();
  const descriptionId = React.useId();
  React.useEffect(() => {
    if (!open) return undefined;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onOpenChange(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onOpenChange]);

  if (!open) return null;
  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/40 p-4 sm:items-center"
      role="dialog"
      aria-modal="true"
      aria-labelledby={title ? titleId : undefined}
      aria-describedby={description ? descriptionId : undefined}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onOpenChange(false);
      }}
    >
      <div className={cn("w-full max-w-lg rounded-md border border-border bg-surface shadow-card", className)}>
        <div className="flex items-start justify-between gap-3 border-b border-border p-4">
          <div className="min-w-0">
            {title && <h2 id={titleId} className="text-sm font-semibold text-text">{title}</h2>}
            {description && <p id={descriptionId} className="mt-0.5 text-xs text-text-muted">{description}</p>}
          </div>
          <button
            type="button"
            aria-label="Close dialog"
            onClick={() => onOpenChange(false)}
            className="shrink-0 text-text-faint hover:text-text"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="p-4">{children}</div>
      </div>
    </div>,
    document.body,
  );
}
